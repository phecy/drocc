## script: find all files in a folder, divide them into 300 batches 
import os 
import tarfile 
import numpy 

## go into one folder and sort the images into segmented batches 
## this is used for one of the tar folders 

maxTar = 30 

# create maxTar tar folders 
folder_filelist=[]
filelists = []
for i in range(maxTar):
    folder_filelist.append([])

folder='imagenet_te' 

# this is the main folder from which data is extracted 
mainpath = '/workplace/wzou/imagenet/imagenet_te/ims/' 

pathlist = os.listdir(mainpath) 

for p in pathlist: 
    folderpath = mainpath+p+'/' 
    
    print 'processing '+folderpath 
    
    list=os.listdir(folderpath) 
    for l in list: 
        fullname = folderpath + l 
        cnt = numpy.random.random_integers(0, maxTar-1, 1)[0]
        #print cnt
        #print len(folder_filelist)
        folder_filelist[cnt].append(fullname) 
    
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
