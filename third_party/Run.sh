#!/bin/bash

# 1. Install openpose on my linux machine
# note: the papers folder is mine

echo -e "Installing openpose...\n"
echo -e "Running openpose config (getting github source)...\n"

git clone  https://github.com/anupamkaul/openpose.git
    cd openpose
    git checkout master
    git remote add upstream https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
    git remote -v
    # to-do (if needed): git pull upstream to master (and git merge with mine for latest updates)

# 2. Building: refer to these guides for building for source for ubuntu:

# https://github.com/anupamkaul/openpose/blob/master/doc/installation/1_prerequisites.md
# https://github.com/anupamkaul/openpose/blob/master/doc/installation/0_index.md







