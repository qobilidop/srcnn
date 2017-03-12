from toolbox.datasets.sr import load_data
from toolbox.models import srcnn
from toolbox.training import train


(x_train, y_train), (x_test, y_test) = load_data()
model = srcnn(x_train.shape[1:], f1=9, f2=1, f3=5)
train(model, x_train, y_train, validation_data=(x_test, y_test),
      nb_epoch=2, resume=True, save_dir='save')
