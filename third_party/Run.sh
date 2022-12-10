#!/bin/bash

# open qn is do I post openpose elsewhere on my git?
# then the pulldown and upstream checks make more sense

# note: the papers folder is mine

echo -e "Running openpose config (getting github source)...\n"

git clone  https://github.com/anupamkaul/openpose.git
    cd openpose
    git remote add upstream https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
    git remote -v
    # git pull upstream to master (and git merge with mine for latest updates)







