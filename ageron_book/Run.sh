#!/bin/bash

echo -e "\nRunning ageron handson-ml 1"

git clone https://github.com/anupamkaul/handson-ml.git

cd handson-ml
  git remote add upstream https://github.com/ageron/handson-ml.git
  git remote -v
  cat howto
      # first time dependencies:
      # conda env create -f environment.yml
      # conda activate tf1
      # python -m ipykernel install --user --name=python3
      # jupyter notebook <.ipynb file> 

  echo -e "\nSimulation of how to start with handson-ml\n"
cd ..





