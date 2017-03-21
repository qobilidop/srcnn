import argparse
from functools import partial
import json
from pathlib import Path

from toolbox.data import load_set
from toolbox.models import get_model
from toolbox.experiment import Experiment


parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=Path)
args = parser.parse_args()
param = json.load(args.param_file.open())

# Model
scale = param['scale']
build_model = partial(get_model(param['model']['name']),
                      **param['model']['params'])

# Data
load_set = partial(load_set,
                   lr_sub_size=param['lr_sub_size'],
                   lr_sub_stride=param['lr_sub_stride'])

# Training
expt = Experiment(scale=param['scale'],
                  load_set=load_set, build_model=build_model,
                  save_dir=param['save_dir'])
expt.train(train_set=param['train_set'], val_set=param['val_set'],
           epochs=param['epochs'], resume=True)

# Evaluation
for test_set in param['test_sets']:
    expt.test(test_set=test_set)
