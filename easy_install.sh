#!/bin/bash

# Install requirements

conda env create -f environment.yaml
conda activate acoustic-anomaly-detection

# Pull data from dvc remote storage
dvc pull
