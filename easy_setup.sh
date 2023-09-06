#!/bin/bash

apt-get update
apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-setuptools python3-wheel \
    git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

micromamba create -y -f environment.yml
micromamba activate acoustic-anomaly-detection
dvc pull
