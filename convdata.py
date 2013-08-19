# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy.random as nr
import numpy as n
import random as r
import tarfile
from PIL import Image
import numpy

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 32
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        
        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
		    #flip the image
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))#flip
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
	    print 'columnumber:',x.shape[1]
	    print 'target shape', target.shape
	    print 'multiview', self.multiview
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
    
class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1

class FlowerDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.border_size = dp_params['crop_border']
        self.inner_size = 160 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
	self.rectify_dup = 1  # test: 5 
        self.inner_crop_dup = 5 * 2  # test: 5 
        self.num_views = self.rectify_dup * self.inner_crop_dup
        #self.num_views = 9*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,160,160))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
    
	
    def __trim_borders(self, x, target):
        y = x.reshape(3, 160, 160, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                   (self.border_size*2, 0), (self.border_size*2, self.border_size*2), 
                                   (0, self.border_size),(self.border_size, 0),
                                   (self.border_size*2, self.border_size), (self.border_size, self.border_size*2)]

                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
		copy_cn = self.inner_crop_dup/2

                for i in xrange(copy_cn):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[: , i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims()  , x.shape[1]))
                    target[: , (copy_cn + i) * x.shape[1]:(copy_cn + i + 1)* x.shape[1]] = pic[: , :  , ::-1 , :].reshape((self.get_data_dims() , x.shape[1]))#flip
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
	    print 'columnumber:',x.shape[1]
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
                
    def get_next_batch(self): 
        epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)
        
	data_copy = self.inner_crop_dup if self.multiview else 1
        cropped  = n.zeros((self.get_data_dims(), datadic['data'].shape[1]*data_copy), dtype=n.single)
        datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1, datadic['data'].shape[1])), (1, data_copy)), requirements='C')
        
        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        #print n.isnan(cropped).sum().sum()
        #print n.isinf(cropped).sum().sum()
        #print n.max(cropped)
        #print n.min(cropped)
        #print datadic
        #print datadic['labels']
        #print n.max(datadic['labels'])
        #print n.min(datadic['labels'])        
        #print len(datadic['labels'])
	#print 'new batch', self.data_mean.shape, 'data:', cropped.shape
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1
    
    def get_batch(self, batch_num):#costomized get batch function
	tar_fid = tarfile.open(self.get_data_file_name(batch_num))
	cn = 0
	data_list = list()
	label_list = list()
	min_side = 160
	while 1:
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
			b = numpy.require(b, requirements='C', dtype=numpy.single)
			
#		        print b.shape, cn
			cn = cn + 1
		#	if cn > 100:
		#		break
			if 2==len(b.shape):
				new_b = numpy.zeros((b.shape[0], b.shape[1], 3), numpy.single)
				new_b[:,:,0] = b;
				new_b[:,:,1] = b;
				new_b[:,:,2] = b;
				b = new_b
			elif 3==len(b.shape) and b.shape[2] != 3:
				continue
			b = numpy.transpose(b, (2, 0, 1))
			#ret_mean = numpy.reshape(ret_mean, (min_side**2 * 3, 1))/( 2 * cn)
			data_list.append(b)
			dir_name = mem.name.split('/')[-2]
			label_list.append(self.batch_meta['class_labels'][dir_name])
		else:
			break
                
	ret_data_cn = len(data_list) * self.rectify_dup
	im_cn = len(data_list)
	data = n.zeros((min_side * min_side * 3 , ret_data_cn)  , dtype=n.single)
	label = n.zeros((1                      , ret_data_cn) , dtype=n.single)
        
	for x, c in zip(data_list, range(0, len(data_list))):
		w = x.shape[2]
		h = x.shape[1]
                
                #print 'width of image:'
                #print w
                #print 'height of image:'
                #print h
                
		#startx = n.floor((w - 160)/2)
		#starty = n.floor((h - 160)/2)
                
                #off_pix = n.floor(startx/3)
                
                #shiftx = n.floor(startx/3) if not startx==0 else 0
                #shifty = n.floor(starty/3) if not starty==0 else 0
                
		startx = (w - 160)/2
		starty = (h - 160)/2
                
		#tmp = x[:, starty:starty + 160, startx:startx + 160]
		#data[:,c] = n.reshape(tmp, (min_side * min_side * 3, ))
		#label[:,c] = label_list[c]
                
		off_pix = 8 
                
		shiftx = off_pix if startx > 3*off_pix else startx 
		shifty = off_pix if starty > 3*off_pix else starty 
                
                start_positions = [(starty,startx), 
				   (starty - shifty, startx - shiftx),
                                   #(starty - 2*shifty, startx - 2*shiftx),
                                   #(starty - 3*shifty, startx - 3*shiftx),
                                   (starty + shifty, startx + shiftx),
                                   #(starty + 2*shifty, startx + 2*shiftx),
                                   #(starty + 3*shifty, startx + 3*shiftx),
                                   (starty - shifty, startx + shiftx),
                                   #(starty - 2*shifty, startx + 2*shiftx),
                                   #(starty - 3*shifty, startx + 3*shiftx),
                                   (starty + shifty, startx - shiftx)]
                                   #(starty + 2*shifty, startx - 2*shiftx),
                                   #(starty + 3*shifty, startx - 3*shiftx)]
                
                end_positions = [(sy+160, sx+160) for (sy,sx) in start_positions]
                #print start_positions
                #print end_positions
                #print x.shape
                #print shiftx
                #print shifty
                #print startx
                #print starty
                for i in xrange(self.rectify_dup):
                    tmp = x[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
		    data  [ :,i * im_cn + c] = n.reshape(tmp, (min_side * min_side * 3, ))
		    label [ :,i * im_cn + c] = label_list[ c]
        
        rnd_seq = numpy.random.permutation(data.shape[1])
        data = data[:, rnd_seq]         
        label = label[:, rnd_seq]
        
	dic = dict()
	dic['data'] = data
	dic['labels'] = label
	#print label
	#exit(0)
        return dic

class ImageNetDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.border_size = dp_params['crop_border']
        self.inner_size = 256 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
	self.rectify_dup = 1  # test: 5 
        self.inner_crop_dup = 5 * 2  # test: 5 
        self.num_views = self.rectify_dup * self.inner_crop_dup
        #self.num_views = 9*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        self.batches_generated = 0
        print 'testing'
        print self.batch_meta['data_mean'].shape
        self.data_mean = self.batch_meta['data_mean'].reshape((3,256,256))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
    
	
    def __trim_borders(self, x, target):
        y = x.reshape(3, 256, 256, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                   (self.border_size*2, 0), (self.border_size*2, self.border_size*2), 
                                   (0, self.border_size),(self.border_size, 0),
                                   (self.border_size*2, self.border_size), (self.border_size, self.border_size*2)]

                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
		copy_cn = self.inner_crop_dup/2

                for i in xrange(copy_cn):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[: , i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims()  , x.shape[1]))
                    target[: , (copy_cn + i) * x.shape[1]:(copy_cn + i + 1)* x.shape[1]] = pic[: , :  , ::-1 , :].reshape((self.get_data_dims() , x.shape[1]))#flip
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
	    print 'columnumber:',x.shape[1]
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
                
    def get_next_batch(self): 
        epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)
        
	data_copy = self.inner_crop_dup if self.multiview else 1
        cropped  = n.zeros((self.get_data_dims(), datadic['data'].shape[1]*data_copy), dtype=n.single)
        datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1, datadic['data'].shape[1])), (1, data_copy)), requirements='C')
        
        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
	print 'new batch', self.data_mean.shape, 'data:', cropped.shape, 'batch num:', batchnum
        #print '----------- start print testing the data batch -------------'
        #print 'size of cropped image '+str(cropped.shape)
        #print 'number of nans '+str(n.isnan(cropped).sum().sum())
        #print 'number of infs '+str(n.isinf(cropped).sum().sum())
        #print 'max value in the batch '+str(n.max(cropped))
        #print 'min value in the batch '+str(n.min(cropped))
        #print datadic['labels']
        #print 'max value in labels '+str(n.max(datadic['labels']))
        #print 'min value in labels '+str(n.min(datadic['labels']))
        #print 'length of all labels '+str(len(datadic['labels'][0]))
        #print 'number of label 3 '+str((datadic['labels'][0] == 3).sum())
        #print '----------- end print testing -------------'
        #print 'possible labels: '+str(datadic['labels'][0][0])
        #print 'possible labels: '+str(datadic['labels'][0][128])
        #print 'possible labels: '+str(datadic['labels'][0][256])
        
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1
    
    def get_batch(self, batch_num):#costomized get batch function
	tar_fid = tarfile.open(self.get_data_file_name(batch_num))
	cn = 0
	data_list = list()
	label_list = list()
	min_side = 256
	while 1:
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
			b = numpy.require(b, requirements='C', dtype=numpy.single)
			
#		        print b.shape, cn
			cn = cn + 1
		#	if cn > 100:
		#		break
			if 2==len(b.shape):
				new_b = numpy.zeros((b.shape[0], b.shape[1], 3), numpy.single)
				new_b[:,:,0] = b;
				new_b[:,:,1] = b;
				new_b[:,:,2] = b;
				b = new_b
			elif 3==len(b.shape) and b.shape[2] != 3:
				continue
			b = numpy.transpose(b, (2, 0, 1))
			#ret_mean = numpy.reshape(ret_mean, (min_side**2 * 3, 1))/( 2 * cn)
			data_list.append(b)
			dir_name = mem.name.split('/')[-2]
			label_list.append(self.batch_meta['class_labels'][dir_name])
		else:
			break
                
	ret_data_cn = len(data_list) * self.rectify_dup
	im_cn = len(data_list)
	data = n.zeros((min_side * min_side * 3 , ret_data_cn)  , dtype=n.single)
	label = n.zeros((1                      , ret_data_cn) , dtype=n.single)
        
	for x, c in zip(data_list, range(0, len(data_list))):
		w = x.shape[2]
		h = x.shape[1]
                
                #print 'width of image:'
                #print w
                #print 'height of image:'
                #print h
                
		#startx = n.floor((w - 256)/2)
		#starty = n.floor((h - 256)/2)
                
                #off_pix = n.floor(startx/3)
                
                #shiftx = n.floor(startx/3) if not startx==0 else 0
                #shifty = n.floor(starty/3) if not starty==0 else 0
                
		startx = (w - 256)/2
		starty = (h - 256)/2
                
		#tmp = x[:, starty:starty + 256, startx:startx +256]
		#data[:,c] = n.reshape(tmp, (min_side * min_side * 3, ))
		#label[:,c] = label_list[c]
                
		off_pix = 8 
                
		shiftx = off_pix if startx > 3*off_pix else startx 
		shifty = off_pix if starty > 3*off_pix else starty 
                
                start_positions = [(starty,startx), 
				   (starty - shifty, startx - shiftx),
                                   #(starty - 2*shifty, startx - 2*shiftx),
                                   #(starty - 3*shifty, startx - 3*shiftx),
                                   (starty + shifty, startx + shiftx),
                                   #(starty + 2*shifty, startx + 2*shiftx),
                                   #(starty + 3*shifty, startx + 3*shiftx),
                                   (starty - shifty, startx + shiftx),
                                   #(starty - 2*shifty, startx + 2*shiftx),
                                   #(starty - 3*shifty, startx + 3*shiftx),
                                   (starty + shifty, startx - shiftx)]
                                   #(starty + 2*shifty, startx - 2*shiftx),
                                   #(starty + 3*shifty, startx - 3*shiftx)]
                
                end_positions = [(sy+256, sx+256) for (sy,sx) in start_positions]
                #print start_positions
                #print end_positions
                #print x.shape
                #print shiftx
                #print shifty
                #print startx
                #print starty
                for i in xrange(self.rectify_dup):
                    tmp = x[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
		    data  [ :,i * im_cn + c] = n.reshape(tmp, (min_side * min_side * 3, ))
		    label [ :,i * im_cn + c] = label_list[ c]
        
        
        #h = numpy.random.randint(10) 
        #numpy.random.seed(h)        
        
        rnd_seq = numpy.random.permutation(data.shape[1])
        data = data[:, rnd_seq]         
        label = label[:, rnd_seq]
        
        #b = list() 
        #b.append(numpy.random.permutation(label[0]).tolist()) 
        #label = b 
        #print label
        
	#exit(0)
        
	dic = dict()
        
	dic['data'] = data
	dic['labels'] = label
        
        return dic

class FoodDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.border_size = dp_params['crop_border']
        self.inner_size = 160 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
	self.rectify_dup = 1
	self.inner_crop_dup = 5 * 2
        #self.num_views = 9*2
        self.num_views = self.rectify_dup * self.inner_crop_dup
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,160,160))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
    
	
    def __trim_borders(self, x, target):
        y = x.reshape(3, 160, 160, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2), 
				  (0, self.border_size),(self.border_size, 0),
				  (self.border_size*2, self.border_size), (self.border_size, self.border_size*2)]

                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
		copy_cn = self.inner_crop_dup/2

                for i in xrange(copy_cn):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[: , i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims()  , x.shape[1]))
                    target[: , (copy_cn + i) * x.shape[1]:(copy_cn + i + 1)* x.shape[1]] = pic[: , :  , ::-1 , :].reshape((self.get_data_dims() , x.shape[1]))#flip
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
	    print 'columnumber:',x.shape[1]
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
    def get_next_batch(self): 
        epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)
        
	data_copy = self.inner_crop_dup if self.multiview else 1
        cropped  = n.zeros((self.get_data_dims(), datadic['data'].shape[1]*data_copy), dtype=n.single)
        datadic['labels'] = n.require(n.tile(datadic['labels'].reshape((1, datadic['data'].shape[1])), (1, data_copy)), requirements='C')
        
        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
	print 'new batch', self.data_mean.shape, 'data:', cropped.shape
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1
    def get_batch(self, batch_num):#costomized get batch function
	tar_fid = tarfile.open(self.get_data_file_name(batch_num))
	cn = 0
	data_list = list()
	label_list = list()
	min_side = 160
	while 1:
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
			b = numpy.require(b, requirements='C', dtype=numpy.single)
			
#		        print b.shape, cn
			cn = cn + 1
		#	if cn > 100:
		#		break
			if 2==len(b.shape):
				new_b = numpy.zeros((b.shape[0], b.shape[1], 3), numpy.single)
				new_b[:,:,0] = b;
				new_b[:,:,1] = b;
				new_b[:,:,2] = b;
				b = new_b
			elif 3==len(b.shape) and b.shape[2] != 3:
				continue
			b = numpy.transpose(b, (2, 0, 1))
			#ret_mean = numpy.reshape(ret_mean, (min_side**2 * 3, 1))/( 2 * cn)
			data_list.append(b)
			dir_name = mem.name.split('/')[-2]
			label_list.append(self.batch_meta['class_labels'][dir_name])
		else:
			break

	ret_data_cn = len(data_list) * self.rectify_dup
	im_cn = len(data_list)
	data = n.zeros((min_side * min_side * 3 , ret_data_cn)  , dtype=n.single)
	label = n.zeros((1                      , ret_data_cn) , dtype=n.single)

	for x, c in zip(data_list, range(0, len(data_list))):
		w = x.shape[2]
		h = x.shape[1]
		startx = (w - 160)/2
		starty = (h - 160)/2

		#tmp = x[:, starty:starty + 160, startx:startx + 160]
		#data[:,c] = n.reshape(tmp, (min_side * min_side * 3, ))
		#label[:,c] = label_list[c]
		
		off_pix = 8
		shiftx = off_pix if startx > off_pix else startx
		shifty = off_pix if starty > off_pix else starty
			

                start_positions = [(starty,startx),  
				   (starty - shifty, startx - shiftx),
                                   (starty + shifty, startx + shiftx),
                                   (starty - shifty, startx + shiftx),
                                   (starty + shifty, startx - shiftx)]

                end_positions = [(sy+160, sx+160) for (sy,sx) in start_positions]
                for i in xrange(self.rectify_dup):
                    tmp = x[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
		    data  [ :,i * im_cn + c] = n.reshape(tmp, (min_side * min_side * 3, ))
		    label [ :,i * im_cn + c] = label_list[ c]

	dic = dict()
	dic['data'] = data
	dic['labels'] = label
	#print label
	#exit(0)
        return dic
