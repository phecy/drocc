#!/bin/bash 

# testing on half imagenet network 
python convnet.py -f ./hondacar_savemodels/./hondacar_savemodels/ConvNet__2013-08-12_00.40.39 --multiview-test=1 --test-only=1 --logreg-name=logprob --test-range=50-51 --num-gpus=1 --test-one 0 #--gpu=2 
