from pathlib import Path

from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import pandas as pd


def train(model, x, y, batch_size=32, epochs=1, validation_data=None,
          save_dir='save', resume=True):
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check architecture
    config_file = save_dir / 'config.yaml'
    if resume and config_file.exists():
        # Check architecture consistency
        saved_model = model_from_yaml(config_file.read_text())
        if model.get_config() != saved_model.get_config():
            raise ValueError('Model architecture has changed.')
    else:
        # Save architecture
        config_file.write_text(model.to_yaml())

    # Set up callbacks
    callbacks = []
    weights_file_template = 'model.{epoch:04d}.hdf5'
    callbacks += [ModelCheckpoint(str(save_dir / weights_file_template))]
    history_file = save_dir / 'history.csv'
    callbacks += [CSVLogger(str(history_file), append=resume)]

    # Inherit weights
    if resume and history_file.exists():
        initial_epoch = pd.read_csv(str(history_file))['epoch'].iloc[-1] + 1
        initial_epoch = int(round(initial_epoch))
    else:
        initial_epoch = 0
    weights_file = save_dir / \
                   weights_file_template.format(epoch=initial_epoch - 1)
    if weights_file.exists():
        model.load_weights(str(weights_file))

    # Train
    model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
              validation_data=validation_data, initial_epoch=initial_epoch)
