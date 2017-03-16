# Super-Resolution Convolutional Neural Network

This is our course project to reproduce and try to improve the Super-Resolution Convolutional Neural Network (SRCNN) from the paper [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092).

## Installation

A Python package `toolbox` is developed to facilitate our experiments. To install it, simply do
 
```bash
pip install -e .
```

The editable mode `-e` is a must to make the data files available.

### Install Dependencies

Dependencies are not specified in `setup.py`, so they need to be installed separately. Two conda environment files are provided. If [GPU prerequisites for TensorFlow](https://www.tensorflow.org/install/install_sources#optional_install_tensorflow_for_gpu_prerequisites) are satisfied, use the GPU version

```bash
conda env create -f environment-gpu.yml
```

Otherwise, use the CPU version

```bash
conda env create -f environment.yml
```

### Set Up Environment on a Vanilla Ubuntu Machine

We need to run our experiments on (nearly) vanilla Ubuntu instances on AWS. So we made it as simple as a single command to set up the environment. 

For the GPU version

```bash
eval "$(curl -fsSL https://raw.githubusercontent.com/qobilidop/srcnn/master/scripts/create-env-gpu.sh)"
```

For the CPU version

```bash
eval "$(curl -fsSL https://raw.githubusercontent.com/qobilidop/srcnn/master/scripts/create-env.sh)"
```

After the execution, you will be in the `~/srcnn` directory and the `srcnn` conda environment, ready to run any experiments.

## Examples

Check the [examples directory](examples).

## Authors

The Deep Glasses Team
* [Zhe An](https://github.com/JasonAn)
* [Bili Dong](https://github.com/qobilidop)
* [Zheng Fang](https://github.com/Catus61)
* [Jiacong Li](https://github.com/jiacong1990)
* [Liyang Xiong](https://github.com/xiongliyang219)

## See Also

* [References](https://github.com/qobilidop/srcnn/wiki/References)
* Other [SRCNN repositories on GitHub](https://github.com/search?utf8=%E2%9C%93&q=srcnn)
