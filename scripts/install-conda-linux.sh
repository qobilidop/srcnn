#!/usr/bin/env bash
# This script installs conda silently on a Linux machine.
# See http://conda.pydata.org/docs/help/silent.html#linux-and-os-x

BASH_RC=$HOME/.bashrc
PREFIX=$HOME/miniconda

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $PREFIX
rm ~/miniconda.sh
echo "
# added by Miniconda3 installer
export PATH=\"$PREFIX/bin:\$PATH\"" >> $BASH_RC
source $BASH_RC
