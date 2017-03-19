"""Example experiment."""
from functools import partial

from toolbox.data import load_set
from toolbox.models import compile
from toolbox.models import fsrcnn
from toolbox.experiment import FSRCNNExperiment


# Model
scale = 3
model = compile(fsrcnn(c=1, d=56, s=12, m=4, k=3))
model.summary()

# Data
train_set = '91-image'
val_set = 'Set5'
test_sets = ['Set5', 'Set14']
load_set = partial(load_set, sub_size=20, sub_stride=100, scale=scale)

# Training
experiment = FSRCNNExperiment(scale=scale, model=model, load_set=load_set,
                              save_dir='.')
experiment.train(train_set=train_set, val_set=val_set, epochs=2, resume=True)

# Evaluation
for test_set in test_sets:
    experiment.test(test_set=test_set)
