"""
@author : arjun-krishna
@desc : Data Management Class, provides batches etc.,
"""
from mnist_reader import *
import numpy as np
from sklearn import preprocessing

TRAIN_DATA_FILE  = '../data/train-images.idx3-ubyte'
TRAIN_LABEL_FILE = '../data/train-labels.idx1-ubyte'

TEST_DATA_FILE  = '../data/t10k-images.idx3-ubyte'
TEST_LABEL_FILE = '../data/t10k-labels.idx1-ubyte'


class DataHandler :

	def __init__(self) :
		self.train_data  = extract_data(TRAIN_DATA_FILE)
		self.scaler = preprocessing.StandardScaler().fit(self.train_data)
		self.train_label = extract_labels(TRAIN_LABEL_FILE)
		self.test_data   = extract_data(TEST_DATA_FILE)
		self.test_label  = extract_labels(TEST_LABEL_FILE)
		self.BATCH_SIZE  = 64
		self.TEST_BATCH_SIZE = 20

	def get_train_batch(self) :
		random_probes = np.random.choice(len(self.train_data), self.BATCH_SIZE, replace=False)
		batch_data  = map(lambda id: self.train_data[id].T,random_probes)
		batch_label = map(lambda id: self.train_label[id],random_probes)
		return batch_data, batch_label

	def get_test_batch(self) :
		random_probes = np.random.choice(len(self.test_data), self.TEST_BATCH_SIZE, replace=False)
		batch_data  = map(lambda id: self.test_data[id].T,random_probes)
		batch_label = map(lambda id: self.test_label[id],random_probes)
		return batch_data, batch_label

	def get_train_data(self) :
		return self.train_data, self.train_label

	def get_test_data(self) :
		return self.test_data, self.test_label

	def get_fixed_test(self) :
		probes = [3, 10, 2, 5, 1, 35, 18, 30, 4, 6, 8, 15, 11, 21, 0, 17, 61, 84, 7, 9 ]
		batch_data  = map(lambda id: self.test_data[id].T,probes)
		batch_label = map(lambda id: self.test_label[id],probes)
		return batch_data, batch_label		