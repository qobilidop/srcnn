from pathlib import Path
import time

from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox.data import load_image_pair
from toolbox.image import array_to_img
from toolbox.image import bicubic_resize
from toolbox.metrics import psnr
from toolbox.paths import data_dir
from toolbox.utils import identity
from toolbox.utils import tf_eval


class Experiment(object):
    def __init__(self, scale=3, model=None, preprocess=identity, load_set=None,
                 save_dir='.'):
        self.scale = scale
        self.model = model
        self.preprocess = preprocess or (lambda x: x)
        self.load_set = load_set
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.save_dir / 'config.yaml'
        self.model_file = self.save_dir / 'model.hdf5'

        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history_file = self.train_dir / 'history.csv'
        self.weights_dir = self.train_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.weights_dir / 'ep{epoch:04d}.hdf5'
        else:
            return self.weights_dir / f'ep{epoch:04d}.hdf5'

    @property
    def latest_epoch(self):
        try:
            return pd.read_csv(str(self.history_file))['epoch'].iloc[-1]
        except FileNotFoundError or pd.io.common.EmptyDataError:
            pass
        return -1

    def train(self, train_set='91-image', val_set='Set5', epochs=1,
              resume=True):
        # Check architecture
        if resume and self.config_file.exists():
            # Check architecture consistency
            saved_model = model_from_yaml(self.config_file.read_text())
            if self.model.get_config() != saved_model.get_config():
                raise ValueError('Model architecture has changed.')
        else:
            # Save architecture
            self.config_file.write_text(self.model.to_yaml())

        # Set up callbacks
        callbacks = []
        callbacks += [ModelCheckpoint(str(self.model_file))]
        callbacks += [ModelCheckpoint(str(self.weights_file()),
                                      save_weights_only=True)]
        callbacks += [CSVLogger(str(self.history_file), append=resume)]

        # Inherit weights
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch)
                self.model.load_weights(str(weights_file))
            initial_epoch = latest_epoch + 1
        else:
            initial_epoch = 0

        # Load data and train
        x_train, y_train = self.load_set(train_set)
        x_val, y_val = self.load_set(val_set)
        self.model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks,
                       validation_data=(x_val, y_val),
                       initial_epoch=initial_epoch)

        # Plot metrics history
        prefix = str(self.history_file).rsplit('.', maxsplit=1)[0]
        df = pd.read_csv(str(self.history_file))
        epoch = df['epoch']
        for metric in ['Loss', 'PSNR']:
            train = df[metric.lower()]
            val = df['val_' + metric.lower()]
            plt.figure()
            plt.plot(epoch, train, label='train')
            plt.plot(epoch, val, label='val')
            plt.legend(loc='best')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig('.'.join([prefix, metric.lower(), 'png']))
            plt.close()

    def test(self, test_set='Set5', metrics=[psnr]):
        print('Test on', test_set)
        image_dir = self.test_dir / test_set
        image_dir.mkdir(exist_ok=True)

        # Evaluate metrics on each image
        rows = []
        for image_path in (data_dir / test_set).glob('*'):
            rows += [self.test_on_image(str(image_path),
                                        str(image_dir / image_path.stem),
                                        metrics=metrics)]
        df = pd.DataFrame(rows)

        # Compute average metrics
        row = pd.Series()
        row['name'] = 'average'
        for col in df:
            if col != 'name':
                row[col] = df[col].mean()
        df = df.append(row, ignore_index=True)

        df.to_csv(str(self.test_dir / f'metrics_{test_set}.csv'))

    def test_on_image(self, path, prefix, suffix='png', metrics=[psnr]):
        lr_image, hr_image = load_image_pair(path, scale=self.scale)
        model = self.model
        x = img_to_array(self.preprocess(lr_image))
        x = x[np.newaxis, :, :, 0:1]

        # Measure run time
        start = time.perf_counter()
        y_pred = model.predict_on_batch(x)
        end = time.perf_counter()

        # Generate output image
        bicubic_image = bicubic_resize(lr_image, self.scale)
        output_array = img_to_array(bicubic_image)
        output_array[:, :, 0] = np.clip(y_pred[0, :, :, 0], 0, 255)
        output_image = array_to_img(output_array, mode='YCbCr')

        # Record metrics
        row = pd.Series()
        row['name'] = Path(path).stem
        row['time'] = end - start
        y_true = img_to_array(hr_image)[np.newaxis, ...]
        for metric in metrics:
            row[metric.__name__] = tf_eval(metric(y_true, y_pred))

        # Save images
        images_to_save = []
        images_to_save += [(hr_image, 'original')]
        images_to_save += [(bicubic_image, 'bicubic')]
        images_to_save += [(output_image, 'output')]
        images_to_save += [(lr_image, 'input')]
        for img, label in images_to_save:
            img.convert(mode='RGB').save('.'.join([prefix, label, suffix]))

        return row
