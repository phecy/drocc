## script to parse the filenames of the ImageNet batches renaming them to the correct format 

import os 

path = '/workplace/' 

list = os.listdir(path)

for l in list:
    if 'data_batch' in l:
        num = l.split('_')[2].strip('0')
        name = 'data_batch_'+num+'.tar'
        command = 'mv '+path+l+' '+path+name
        print command 
        os.system(command)
