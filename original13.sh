#python convnet.py --data-path=/workplace/wzou/cuda-convnet/data/ --save-path=/workplace/wzou/cuda-convnet/tmp --test-range=5 --train-range=1-4 --layer-def=./example-layers/layers-conv-local-13pct.cfg --layer-params=./example-layers/layer-params-conv-local-13pct.cfg --gpu=1  --data-provider=cifar-cropped --test-freq=13 --crop-border=4 --epochs=100 

#python convnet.py -f tmp/ConvNet__2013-07-30_16.05.06 --train-range=1-5 --epochs=140 

python convnet.py -f /workplace/wzou/cuda-convnet/tmp/ConvNet__2013-07-30_16.05.06 --train-range=1-5 --test-range=6 --epochs=160 
