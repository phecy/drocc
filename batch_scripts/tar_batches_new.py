## script: find all files in a folder, divide them into 300 batches 
import os 
import tarfile 
import numpy 

## go into one folder and sort the images into segmented batches 
## this is used for one of the tar folders 

maxTar = 400 

# create maxTar tar folders 
folder_filelist=[]
filelists = []
for i in range(maxTar):
    folder_filelist.append([])

folder='imagenet_tr'

# this is the main folder from which data is extracted 
mainpath = '/mnt/exthdd/imagenet_data/imagenet/'+folder+'/' 

pathlist = os.listdir(mainpath) 

all_file_list = [] 

for p in pathlist: 
    folderpath = mainpath+p+'/' 
    print 'processing '+folderpath     
    list=os.listdir(folderpath) 
    for l in list: 
        fullname = folderpath + l 
        all_file_list.append(fullname) 

all_file_list = numpy.random.permutation(all_file_list)

for f in all_file_list: 
    cnt = numpy.random.random_integers(0, maxTar-1, 1)[0] 
    folder_filelist[cnt].append(f)

for i in range(maxTar): 
    print i
    filename = folder+'/data_batch_'+str(i+1)+'.txt' 
    curfile = open(filename, 'w')
    for f in folder_filelist[i]:
        #tarfolders[i].add(f)
        curfile.write(f+'\n')
    #tarfolders[i].close() 
    curfile.close()

#put in one of the tar folders 
#path = '/workplace/' 
#list = os.listdir(path) 
#for l in list: 
#    if 'data_batch' in l: 
#        num = l.split('_')[2].strip('0') 
#        name = 'data_batch_'+num+'.tar' 
#        command = 'mv '+path+l+' '+path+name 
#        print command 
#        os.system(command) 
