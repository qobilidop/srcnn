from pathlib import Path

from keras.models import load_model
from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
import numpy as np
import pandas as pd

from toolbox.preprocessing import array_to_img
from toolbox.preprocessing import bicubic_resize
from toolbox.preprocessing import modcrop
from toolbox.metrics import psnr


class Experiment(object):
    def __init__(self, save_dir='save'):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.config_file = save_dir / 'config.yaml'
        self.history_file = save_dir / 'history.csv'
        self.model_file = save_dir / 'model.hdf5'

    @property
    def model(self):
        return load_model(str(self.model_file), custom_objects={'psnr': psnr})

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.save_dir / 'model.{epoch:04d}.hdf5'
        else:
            return self.save_dir / f'model.{epoch:04d}.hdf5'

    def train(self, model, x, y, batch_size=32, epochs=1, validation_data=None,
              resume=True):
        # Check architecture
        if resume and self.config_file.exists():
            # Check architecture consistency
            saved_model = model_from_yaml(self.config_file.read_text())
            if model.get_config() != saved_model.get_config():
                raise ValueError('Model architecture has changed.')
        else:
            # Save architecture
            self.config_file.write_text(model.to_yaml())

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
            model.load_weights(str(weights_file))

        # Train
        model.fit(x, y, batch_size=batch_size, epochs=epochs,
                  callbacks=callbacks, validation_data=validation_data,
                  initial_epoch=initial_epoch)

    def test_on_image(self, image_path, prefix, suffix='png'):
        scale = 3
        image = load_img(image_path,)
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
