"""
@author : arjun-krishna
@desc : Read the byte encoded MNIST data in Lecun's page
"""
from __future__ import print_function

import struct
import numpy as np
from PIL import Image

"""
display flattended image with (r,c) dimension
"""
def display_img(img, r, c, file=None) :
	img = img.reshape(r,c)
	disp = Image.fromarray(img)
	if file :
		disp.save(file)
	else :
		disp.show()

"""
output : List of flattended images
"""
def extract_data(filename) :
	print ('Extracting data from', filename.split('/')[-1])
	print ('-------------------------------------------------')
	data = []
	with open(filename, 'r') as bytestream :

		MAGIC_NUM = struct.unpack('>i', bytestream.read(4))[0]
		NUM_IMAGES = struct.unpack('>i', bytestream.read(4))[0]
		NUM_ROW = struct.unpack('>i', bytestream.read(4))[0]
		NUM_COL = struct.unpack('>i', bytestream.read(4))[0]
		
		print ('Number of Images : ', NUM_IMAGES)
		for i in xrange(NUM_IMAGES) :
			mssg = "Loading [{0:0.2f}%]".format(float((i+1)*100)/NUM_IMAGES)
			clear = "\b"*(len(mssg))
			print(mssg, end="")
			buf = bytestream.read(NUM_ROW*NUM_COL)
			img = np.frombuffer(buf, dtype=np.uint8)
			# img = img.reshape(NUM_ROW, NUM_COL)
			data.append(img)
			print(clear, end="")

	print ('\nExtraction Completed!')
	print ('-------------------------------------------------')
	return data

"""
output : List of labels
"""
def extract_labels(filename) :
	print ('Extracting Labels from', filename.split('/')[-1])
	print ('-------------------------------------------------')
	data = []
	with open(filename, 'r') as bytestream :

		MAGIC_NUM = struct.unpack('>i', bytestream.read(4))[0]
		NUM_ITEMS = struct.unpack('>i', bytestream.read(4))[0]
		
		print ('Number of Items : ', NUM_ITEMS)
		for i in xrange(NUM_ITEMS) :
			mssg = "Loading [{0:0.2f}%]".format(float((i+1)*100)/NUM_ITEMS)
			clear = "\b"*(len(mssg))
			print(mssg, end="")
			label = struct.unpack('>B', bytestream.read(1))[0]
			data.append(label)
			print(clear, end="")

	print ('\nExtraction Completed!')
	print ('-------------------------------------------------')
	return data
