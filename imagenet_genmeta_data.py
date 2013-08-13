## ok what does this script do? 
## generates 
## - mean image 
## - labels/some other information 
## as meta data for the dataset 

from data import *
from imagenet_get_meanIm import *
class_dict_fn = 'imagenet_class_name_label.txt'
batch_meta = dict()

class_dict = dict()
for line in open(class_dict_fn, 'r'):
	line = line.strip('\n')
	line = line.split()
	class_dict[line[0]] = int(line[1])
batch_meta['class_labels'] = class_dict
batch_meta['label_names'] = class_dict.keys()
batch_meta['dp_type'] = 'ImageNetDataProvider'
#batch_meta['data_in_rows'] = True
batch_meta['num_vis'] = 224*224*3 #49152

tar_fn_list = '/workplace/wzou/imagenet/imagenet_tr/tar_files.txt'

cn = 0
for line in open(tar_fn_list, 'r'):
	line = line.strip('\n')
	if  cn == 0:
		mean_im = GetMeanImage(line, 256, 4)
	else:
		mean_im = mean_im + GetMeanImage(line, 256, 4)
	cn = cn + 1
	print cn
mean_im = mean_im/cn
mean_im = numpy.require(mean_im, requirements='C', dtype=numpy.single)
batch_meta['data_mean'] = mean_im
#pickle(BATCH_META_FILE, batch_meta)
pickle(BATCH_META_FILE + '.tmp', batch_meta)
