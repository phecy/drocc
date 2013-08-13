import tarfile
from PIL import Image
import numpy

def GetMeanImage(tar_fn, min_side=256, skip=4):
	ret_mean = numpy.zeros((3, min_side, min_side), dtype =numpy.double)
	tar_fid = tarfile.open(tar_fn)
	cn = 0
	while 1:
		for dummy in range(10):
			mem = tar_fid.next()
		if mem != None:
			try:
				img = Image.open(tar_fid.extractfile(mem))
			except:
				continue

			w = img.size[0]
			h = img.size[1]
			if w < h: 
				size = min_side, int(min_side * h/w)
				max_side =  int(min_side * h/w)
			else:
				size = int(min_side * w/h), min_side
				max_side =  int(min_side * w/h)

			img = img.resize(size)

			b = (numpy.asarray(img))
			b = numpy.require(b, requirements='C', dtype=numpy.double)
			
			if 2==len(b.shape):
				new_b = numpy.zeros((b.shape[0], b.shape[1], 3), numpy.double)
				new_b[:,:,0] = b;
				new_b[:,:,1] = b;
				new_b[:,:,2] = b;
				b = new_b
			b = numpy.transpose(b, (2, 0, 1))

			border = max_side - min_side
			if b.shape[0] != 3:
				continue
			for offset in range(0, border, skip):
				cn = cn + 1
				if w < h:
					ret_mean = ret_mean + b[:, offset:offset + min_side, :]
				else:
					ret_mean = ret_mean + b[:, :, offset:offset + min_side]
				



		else:
			break
	ret_mean = ret_mean + ret_mean[:, :, ::-1]#flip the image
	ret_mean = numpy.reshape(ret_mean, (min_side**2 * 3, 1))/( 2 * cn)
#	print 'tot cropped images:', cn
#	print ret_mean
	return ret_mean

#if __name__ == "__main__": 
#	GetMeanImage('/workplace/xiaoyu/flower_data/train/data_batch_0.tar', 256, 4)
