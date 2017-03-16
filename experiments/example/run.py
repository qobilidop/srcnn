"""Example experiment."""
from toolbox.models import compile_srcnn
from toolbox.experiment import Experiment


model = compile_srcnn()
model.summary()
experiment = Experiment(model, save_dir='.')
experiment.train(epochs=1)
experiment.train(epochs=2, resume=True)
experiment.test()
