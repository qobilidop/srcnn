from pathlib import Path

from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import pandas as pd


class Experiment(object):
    def __init__(self, save_dir='save'):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save = save_dir
        self.config_file = save_dir / 'config.yaml'
        self.history_file = save_dir / 'history.csv'

    def weights_file(self, epoch=None):
        if epoch is None:
            return self.save / 'model.{epoch:04d}.hdf5'
        else:
            return self.save / f'model.{epoch:04d}.hdf5'

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
        callbacks += [ModelCheckpoint(str(self.weights_file()))]
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
