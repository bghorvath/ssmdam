#!/bin/bash

apt-get update
apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-setuptools python3-wheel \
    git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev apt-transport-https

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update
apt-get install google-cloud-cli

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source /root/.bashrc

micromamba create -y -f environment.yml
micromamba activate acoustic-anomaly-detection

gcloud auth application-default login
dvc pull
