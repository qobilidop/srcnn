import os

from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import pandas as pd


def train(model, x, y, batch_size=32, nb_epoch=1, validation_data=None,
          optimizer='adam', loss='mse', metrics=[],
          resume=False, save_dir='save'):
    os.makedirs(save_dir, exist_ok=True)
    model_file = os.path.join(save_dir, 'model.hdf5')
    history_file = os.path.join(save_dir, 'history.csv')

    if resume and os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model.compile(optimizer, loss, metrics=metrics)

    # set up callbacks to record model and history
    model_checkpoint = ModelCheckpoint(model_file)
    csv_logger = CSVLogger(history_file, append=resume)

    # determine initial epoch
    if resume and os.path.exists(history_file):
        initial_epoch = pd.read_csv('save/history.csv')['epoch'].iloc[-1] + 1
    else:
        initial_epoch = 0

    model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,
              callbacks=[model_checkpoint, csv_logger],
              validation_data=validation_data,
              initial_epoch=initial_epoch)
