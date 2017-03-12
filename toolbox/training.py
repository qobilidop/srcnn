import os

from keras.models import model_from_yaml
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import pandas as pd


def train(model, x, y, batch_size=32, nb_epoch=1, validation_data=None,
          optimizer='adam', loss='mse', metrics=[],
          save_dir='save', resume=True):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Check architecture
    config_file = os.path.join(save_dir, 'config.yaml')
    if resume and os.path.exists(config_file):
        # Check architecture consistency
        saved_model = model_from_yaml(open(config_file).read())
        if model.get_config() != saved_model.get_config():
            raise ValueError('Model architecture has changed.')
    else:
        # Save architecture
        open(config_file, 'w').write(model.to_yaml())

    # Set up loggers
    weights_file_template = os.path.join(save_dir, 'weights.{epoch:04d}.hdf5')
    model_checkpoint = ModelCheckpoint(weights_file_template,
                                       save_best_only=True)
    history_file = os.path.join(save_dir, 'history.csv')
    csv_logger = CSVLogger(history_file, append=resume)

    # Inherit weights
    if resume and os.path.exists(history_file):
        initial_epoch = pd.read_csv('save/history.csv')['epoch'].iloc[-1] + 1
    else:
        initial_epoch = 0
    weights_file = weights_file_template.format(epoch=initial_epoch - 1)
    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    # Train
    model.compile(optimizer, loss, metrics=metrics)
    model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,
              callbacks=[model_checkpoint, csv_logger],
              validation_data=validation_data,
              initial_epoch=initial_epoch)
