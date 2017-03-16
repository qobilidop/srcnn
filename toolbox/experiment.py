from pathlib import Path

from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import pandas as pd

from toolbox.data import load_set
from toolbox.paths import data_dir
from toolbox.preprocessing import array_to_img
from toolbox.preprocessing import bicubic_resize
from toolbox.preprocessing import modcrop


class Experiment(object):
    def __init__(self, model, preprocess=None, save_dir='save'):
        self.model = model
        self.preprocess = preprocess
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.config_file = save_dir / 'config.yaml'
        self.history_file = save_dir / 'history.csv'
        self.model_file = save_dir / 'model.hdf5'

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.save_dir / 'model.{epoch:04d}.hdf5'
        else:
            return self.save_dir / f'model.{epoch:04d}.hdf5'

    def train(self, train_set='91-image', val_set='Set5',
              batch_size=32, epochs=1, resume=True):
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
        if resume and self.history_file.exists():
            try:
                epoch = pd.read_csv(str(self.history_file))['epoch']
                initial_epoch = epoch.iloc[-1] + 1
                initial_epoch = int(round(initial_epoch))
            except pd.io.common.EmptyDataError:
                initial_epoch = 0
        else:
            initial_epoch = 0
        weights_file = self.weights_file(epoch=initial_epoch - 1)
        if weights_file.exists():
            self.model.load_weights(str(weights_file))

        # Load data and train
        x_train, y_train = load_set(train_set)
        x_val, y_val = load_set(val_set)
        if self.preprocess is not None:
            x_train = self.preprocess(x_train)
            x_val = self.preprocess(x_val)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks, validation_data=(x_val, y_val),
                       initial_epoch=initial_epoch)

    def test(self, test_set='Set5'):
        output_dir = self.save_dir / test_set
        output_dir.mkdir(exist_ok=True)
        for image_path in (data_dir / 'Test' / test_set).glob('*'):
            self.test_on_image(str(image_path),
                               str(output_dir / image_path.stem))

    def test_on_image(self, image_path, prefix, suffix='png'):
        scale = 3
        image = load_img(image_path)
        image = image.convert('YCbCr')
        im_label = modcrop(image, scale)
        im_input = bicubic_resize(bicubic_resize(im_label, 1 / scale),
                                  im_label.size)
        im_label = img_to_array(im_label)[6:-6, 6:-6, :]
        im_input = img_to_array(im_input)

        model = self.model
        x = im_input[np.newaxis, :, :, 0:1]
        y = model.predict_on_batch(x)

        output_arr = im_label
        output_arr[:, :, 0] = y[0, :, :, 0]

        arrays_to_save = []
        arrays_to_save += [(img_to_array(image), 'original')]
        arrays_to_save += [(im_label, 'bicubic')]
        arrays_to_save += [(output_arr, 'output')]
        arrays_to_save += [(im_input, 'input')]
        for array, label in arrays_to_save:
            img = array_to_img(array, mode='YCbCr')
            img.convert(mode='RGB').save('.'.join([prefix, label, suffix]))
