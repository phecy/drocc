#!/bin/bash

exp=norm

python convnet.py -f /home/wzou/ccdropout/imagenet_savemodels/ConvNet__2013-07-22_21.42.31-$exp --test-range=390-399 --train-range=0-389 --layer-params=./imagenet-layers/layer-params-conv-local-imagenet-full-256-$exp.cfg 1>/home/wzou/ccdropout/imagenet_savemodels/full_imagenet-$exp.log 2>/home/wzou/ccdropout/imagenet_savemodels/full_imagenet-$exp.err --test-freq=13 --gpu=0 --data-path=/mnt/exthdd/imagenet_data/imagenet_batches/imagenet_tr/ --data-provider=imagenet 
