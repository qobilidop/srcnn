#!/usr/bin/env bash
# This script creates a conda environment on a vanilla Ubuntu machine
# from scratch.

sudo apt-get -y install git
cd ~
git clone -q https://github.com/qobilidop/srcnn.git
cd srcnn
. install/install-conda-on-linux.sh
conda env create -f install/env-gpu.yml
source activate srcnn
pip install -e .
