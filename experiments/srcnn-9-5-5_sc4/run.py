from functools import partial

from toolbox.data import load_set
from toolbox.models import compile
from toolbox.models import srcnn
from toolbox.experiment import Experiment
from toolbox.image import bicubic_resize


# Model
scale = 4
model = compile(srcnn(c=1, f1=9, f2=5, f3=5, n1=64, n2=32))
model.summary()

# Data
train_set = '91-image'
val_set = 'Set5'
test_sets = ['Set5', 'Set14']
preprocess = partial(bicubic_resize, size=scale)
load_set = partial(load_set, sub_size=11, sub_stride=5, scale=scale,
                   channel=0, preprocess=preprocess)

# Training
experiment = Experiment(scale=scale, model=model, preprocess=preprocess,
                        load_set=load_set, save_dir='.')
experiment.train(train_set=train_set, val_set=val_set, epochs=500, resume=True)

# Evaluation
for test_set in test_sets:
    experiment.test(test_set=test_set)
