# Use Keras with Python 3

## Preparation

Make sure there's (much) more than 8G space when launching the AWS instance. The initial file on the instance + miniconda + tensorflow + keras will take around 8G space. Therefore, to ensure there's enough space for data, maximum allowed storage space is recommended when launching a new instance.

## Install Miniconda

Use the following bash script to download the Miniconda (for Python 3 on Linux) installation package.
```
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Install Miniconda by running the installation package.
```
bash Miniconda3-latest-Linux-x86_64.sh
```

Choose yes all the way, until the installation finishes. Then, activate the installation by 
```
source ~/.bashrc
```

You can verify that Miniconda is installed by running a ```conda``` command, e.g.
```
conda list
```

## Setup a Conda Environment Using Python 3
Create a new environment named "my_env" by running
```
conda create --name my_env python=3
```
Activate the new environment by running
```
source activate my_env
```
When you want to stop using this envinronment, run
```
source deactivate
```

## Install TensorFlow and Keras in the Python 3 Environment
Same as usual, i.e.
```
source activate my_env
pip install tensorflow
pip install keras
```

* [Reference](https://www.howtoing.com/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04/)
