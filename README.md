# Super-Resolution Convolutional Neural Network

This is our course project to reproduce and try to improve the Super-Resolution Convolutional Neural Network (SRCNN) from the paper [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092).

## Installation

A Python package `toolbox` is developed to facilitate our experiments. To install it along with all its dependencies, a conda environment is suggested to be created. If conda is not installed and you are on a Linux machine, run
 
 ```bash
. install-conda-linux.sh
```

to install it. If [GPU prerequisites for TensorFlow](https://www.tensorflow.org/install/install_sources#optional_install_tensorflow_for_gpu_prerequisites) are satisfied, use the GPU version

```bash
conda env create -f environment-gpu.yml
```

Otherwise, use the CPU version

```bash
conda env create -f environment.yml
```

The commands above create a conda environment `srcnn`, which could be activated by

```bash
source activate srcnn
```

## Examples

Check the [examples directory](examples).

## Team Members

* [Zhe An](https://github.com/JasonAn)
* [Bili Dong](https://github.com/qobilidop)
* [Zheng Fang](https://github.com/Catus61)
* [Jiacong Li](https://github.com/jiacong1990)
* [Liyang Xiong](https://github.com/xiongliyang219)

## See Also

* [References](https://github.com/qobilidop/srcnn/wiki/References)
* Various other [repositories on GitHub](https://github.com/search?utf8=%E2%9C%93&q=srcnn)
