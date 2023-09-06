#!/bin/bash

# Install requirements
apt-get update
apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-setuptools python3-wheel \
    git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev

# Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

export MAMBA_EXE="$HOME/.local/bin/micromamba";
export MAMBA_ROOT_PREFIX='micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback
fi
unset __mamba_setup

# Create and activate micromamba environment
micromamba create -y -f environment.yml
micromamba activate acoustic-anomaly-detection

# Pull data from dvc remote storage
dvc pull
