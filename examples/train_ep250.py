"""Demonstrate how to train a model."""
from toolbox.datasets.sr import load_data
from toolbox.models import srcnn
from toolbox.training import train


# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Check data shapes
print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

# Build model
model = srcnn(x_train.shape[1:])

# Show model summary
model.summary()

# Start training
train(model, x_train, y_train, validation_data=(x_test, y_test))

# Resume training
train(model, x_train, y_train, validation_data=(x_test, y_test),
      nb_epoch=250, resume=True)
