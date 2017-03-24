# Super-Resolution Convolutional Neural Network

We have implemented [SRCNN], [FSRCNN] and [ESPCN] in [Keras] with [TensorFlow] backend.

[SRCNN]: https://arxiv.org/abs/1501.00092
[FSRCNN]: https://arxiv.org/abs/1608.00367
[ESPCN]: https://arxiv.org/abs/1609.05158
[Keras]: https://github.com/fchollet/keras
[TensorFlow]: https://github.com/tensorflow/tensorflow

## Installation

A Python package `toolbox` is developed to facilitate our experiments. You need to install it to reproduce our experiments. If the dependencies as defined in [env-gpu.yml](install/env-gpu.yml) or [env-cpu.yml](install/env-cpu.yml) are already satisfied, simply do
 
```bash
pip install -e .
```

to install the package. Otherwise you can create a conda environment `srcnn` for all the dependencies by

```bash
conda env create -f install/env-gpu.yml
```

or

```bash
conda env create -f install/env-cpu.yml
```

We have also provided scripts to make it easy to set up an environment on a vanilla Ubuntu machine. Simply do

```bash
eval "$(curl -fsSL https://raw.githubusercontent.com/qobilidop/srcnn/master/install/create-env-gpu.sh)"
```

or

```bash
eval "$(curl -fsSL https://raw.githubusercontent.com/qobilidop/srcnn/master/install/create-env-cpu.sh)"
```

and you'll be in the `~/srcnn` directory and the `srcnn` conda environment, ready to run any experiment.

## Authors

The Deep Glasses Team :eyeglasses:
* [Zhe An](https://github.com/JasonAn)
* [Bili Dong](https://github.com/qobilidop)
* [Zheng Fang](https://github.com/Catus61)
* [Jiacong Li](https://github.com/jiacong1990)
* [Liyang Xiong](https://github.com/xiongliyang219)
