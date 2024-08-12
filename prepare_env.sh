#!/usr/bin/env bash

# Ensure the script stops at any error
set -e

# Start create environment
echo Start create environment!

conda create -n hsic_env python=3.10 -y

conda activate hsic_env

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

conda install matplotlib pandas lightgbm scikit-learn imageio scikit-image -y

conda install -n hsic_env ipykernel --update-deps --force-reinstall -y

echo Finish creating environment!