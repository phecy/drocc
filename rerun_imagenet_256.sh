python convnet.py -f ./imagenet_savemodels/half_imagenet/ConvNet__2013-08-12_00.35.42_copy --save-path=/imagenet_savemodels/half_imagenet/ --epochs=300000 1>imagenet_savemodels/half_imagenet/half_256_cont.log 2>imagenet_savemodels/half_imagenet/half_256_cont.err 

#python convnet.py --data-path=/mnt/exthdd/imagenet_data/imagenet_batches/imagenet_tr/ --save-path=./imagenet_savemodels/half_imagenet/ --test-range=390-399 --train-range=0-389 --layer-def=./imagenet-layers/layers-conv-local-imagenet-half-256.cfg --layer-params=./imagenet-layers/layer-params-conv-local-imagenet-half-256.cfg --data-provider=imagenet --test-freq=13 --crop-border=16 --epochs=300000 1>imagenet_savemodels/half_imagenet/half_256.log 2>imagenet_savemodels/half_imagenet/half_256.err

#1>/workplace/wzou/xy_cudaconvnet/imagenet_savemodels/full_imagenet_new7.log 2>/workplace/wzou/xy_cudaconvnet/imagenet_savemodels/full_imagenet_new7.err 
