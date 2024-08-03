#!/usr/bin/env bash

# Ensure the script stops at any error
set -e

# Start create environment
echo Start create environment!

conda env create -f hsic_env.yaml

conda activate hsic_env

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

echo Finish creating environment!