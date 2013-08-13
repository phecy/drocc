import os

path = '/workplace/imagenet/imagenet_tr/'
list = os.listdir(path)

output = 'imagenet_class_labels.txt'
f = open(output, 'w')

cnt = 0
for l in list: 
    f.write(l+' '+str(cnt)+'\n')
    cnt += 1
