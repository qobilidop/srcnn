# Installation Instructions

A Python package `toolbox` is developed to facilitate our experiments. To install it, simply do
 
```bash
pip install -e .
```

The editable mode `-e` is a must to make the data files available.

## Install Dependencies

Dependencies are not specified in `setup.py`, so they need to be installed separately. Two conda environment files are provided. If [GPU prerequisites for TensorFlow](https://www.tensorflow.org/install/install_sources#optional_install_tensorflow_for_gpu_prerequisites) are satisfied, use the GPU version

```bash
conda env create -f gpu-environment.yml
```

Otherwise, use the CPU version

```bash
conda env create -f cpu-environment.yml
```

## Set Up Environment on a Vanilla Ubuntu Machine

We need to run our experiments on (nearly) vanilla Ubuntu instances on AWS. So we made it as simple as a single command to set up the environment. 

For the GPU version

```bash
eval "$(curl -fsSL https://raw.githubusercontent.com/qobilidop/srcnn/master/install/create-gpu-env.sh)"
```

For the CPU version

```bash
eval "$(curl -fsSL https://raw.githubusercontent.com/qobilidop/srcnn/master/install/create-cpu-env.sh)"
```

After the execution, you will be in the `~/srcnn` directory and the `srcnn` conda environment, ready to run any experiments.
