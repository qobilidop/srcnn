#### Update on 2018-10-07

Just find that fast.ai now has a fantastic [lecture on super resolution](http://course.fast.ai/lessons/lesson14.html). Check it out!

#### Update on 2018-03-30

Beyond our expectation, this 2-week course project got some visitors. Although the project's quality is not to our satisfactory, all the authors have since moved on to other things, with no intention to improve it. Thus we felt obliged to list some useful resources here for anyone who stumbles on our page:

- [huangzehao/Super-Resolution.Benckmark](https://github.com/huangzehao/Super-Resolution.Benckmark)
- [IvoryCandy/super-resolution](https://github.com/IvoryCandy/super-resolution)
- [YapengTian/Single-Image-Super-Resolution](https://github.com/YapengTian/Single-Image-Super-Resolution)

There are much more resources on this topic than listed above. One way to find them is to search `super resolution` on GitHub.

What follows is the original README:

# Convolutional Neural Networks for Single Image Super-Resolution

We have implemented [SRCNN], [FSRCNN] and [ESPCN] in [Keras] with [TensorFlow] backend. The network architectures are implemented in [models.py](toolbox/models.py) and [layers.py](toolbox/layers.py). Our results are described in our [final report](https://github.com/qobilidop/srcnn/releases/download/final/final-report.pdf). The [experiments data](https://github.com/qobilidop/srcnn/releases/download/final/experiments-data.zip) used to get our results are also provided. To reduce the file size, weights files at each epoch are not included in the data file. But the final model file is included and there are enough data to reproduce all the plots in our final report.

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

and you'll be in the ~/srcnn directory and the `srcnn` conda environment, ready to run any experiment.

## Experiments

An experiment is configured by a json file in the [experiments](experiments) directory. To run an experiment, `cd` into the experiments directory and do

```bash
python run.py {experiment-name}.json
```

You can also do

```bash
python run_all.py
```

to run all the experiments at once. But note that it may take a very long time.

Once some experiments are finished, diagnostic plots can be made by

```bash
python plot.py
```

## Authors

The Deep Glasses Team :eyeglasses:
* [Zhe An](https://github.com/JasonAn)
* [Bili Dong](https://github.com/qobilidop)
* [Zheng Fang](https://github.com/Catus61)
* [Jiacong Li](https://github.com/jiacong1990)
* [Liyang Xiong](https://github.com/xiongliyang219)
