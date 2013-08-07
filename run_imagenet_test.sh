#!/bin/bash 

# testing on half imagenet network 
python convnet.py -f /home/ma/wzou/dropout/imagenet_savemodels/ConvNet__2013-07-22_21.42.31 --multiview-test=1 --test-only=1 --logreg-name=logprob --test-range=500-529 --num-gpus=1 --gpu=2 --test-one 0 
