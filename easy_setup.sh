#!/bin/bash

apt-get update

curl https://pyenv.run | bash
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install
pyenv local
pyenv exec pip install --no-cache-dir poetry

poetry install --with gpu
poetry run dvc pull
