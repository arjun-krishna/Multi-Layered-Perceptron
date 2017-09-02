"""
@author : arjun-krishna
@desc :
The MLP class, with the methods to train and test
"""
from __future__ import print_function

import numpy as np
from mnist_reader import display_img

def sigmoid(x, derivative=False) :
	if derivative :
		y = sigmoid(x)
		return y*(1 - y)
	else :
		return 1 / (1 + np.exp(-x))

def relu(x, derivative=False) :
	if derivative :
		return np.maximum(0, np.sign(x))
	else :
		return np.maximum(0, x)

def softmax(x) :
	ex = np.exp((x - np.max(x)))
	return ex / ex.sum(axis=0)

class MLP :
	"""
	config = Is the Network structure | dimensions of x,h1,h2,h3..,y in a list
	activation = List of activation functions for h1, h2,..,y
	
	Example Launch Config :
	NN = MLP([4,6,2],[sigmoid, softmax])
	"""
	def __init__(self, config, activation) :

		self.nl = len(config)																
		self.z = {}
		self.a = {}
		self.nh = self.nl - 2 			 # No. of Hidden Layers
		for i in range(1,self.nl+1) :
			self.z[i] = np.zeros((config[i-1],1))
			self.a[i] = np.zeros((config[i-1],1))
		self.W = {}
		self.b = {}
		self.init_mu    = 0 
		self.init_sigma = 0.08
		for i in range(1,self.nl) :
			self.W[i] = self.init_sigma*np.random.randn(config[i],config[i-1]) + self.init_mu # Random initial weights
			self.b[i] = np.zeros((config[i],1))
		self.config = config
		self.activation = activation

	def forward_pass(self, x) :
		self.a[1] = x
		for i in range(1,self.nl) :
			self.z[i+1] = np.dot(self.W[i],self.a[i]) + self.b[i]
			self.a[i+1] = self.activation[i-1](self.z[i+1])
		return self.a[self.nl]

	"""
	max_epoch - Maximum training epochs
	freq_test_loss - The frequency with which we log the test performance.
	name - The folder (in log dir) in which progress logs will be stored  
	alpha - Learning rate
	lamda - Regularization parameter
	gamma - Momentum parameter
	lrf   - The decay factor
	lr_itr - The iteration after which decay occurs
	"""
	def train_mini_batch(self, data_handler,name='model_1',act='sigmoid', max_epoch=10000, freq_test_loss = 200, alpha=0.01, lamda=0.005, gamma=0.8, lrf=1.0, lr_itr=250) :

		SCALER = data_handler.scaler

		TEST_DATA, TEST_LABELS = data_handler.get_test_data()
		TEST_DATA = SCALER.transform(TEST_DATA)
		TEST_DATA = np.array(TEST_DATA).T
		TEST_SIZE = len(TEST_LABELS)

		BATCH_SIZE = data_handler.BATCH_SIZE
		vW = {}
		vb = {}
		delta = {}

		config_log = open('log/'+name+'/config','w')
		train_loss_log = open('log/'+name+'/train_loss.csv','w')
		test_loss_log = open('log/'+name+'/test_loss.csv','w')
		test_acc_log = open('log/'+name+'/test_acc.csv','w')

		train_loss_log.write('epoch, train_loss\n')
		test_loss_log.write('epoch, test_loss\n')
		test_acc_log.write('epoch, test_accuracy\n')

		config_log.write('Activation = '+act+'\n')
		config_log.write('Alpha  = '+str(alpha)+'\n')
		config_log.write('Lambda = '+str(lamda)+'\n')

		print ('Training the Network')
		print ('-------------------------------------------------')

		for l in range(1,self.nl) :
			vW[l] = np.zeros(self.W[l].shape)
			vb[l] = np.zeros(self.b[l].shape)


		for epoch in range(1, max_epoch+1) :

			mssg = "Training Progress [{}%]".format(float(epoch*100)/max_epoch)
			clear = "\b"*(len(mssg))
			print(mssg, end="")

			if (epoch % lr_itr == 0) :
				alpha = alpha*lrf


			if (epoch % freq_test_loss == 0) :
				test_acc = 0.0
				test_loss = 0.0

				_y = self.forward_pass(TEST_DATA)
				for i in range(TEST_SIZE) :
					if( np.argmax(_y[:,i]) == TEST_LABELS[i] ) :
						test_acc += 1
						test_loss += -np.log(_y[TEST_LABELS[i],i])

				test_loss /= TEST_SIZE

				test_acc_log.write(str(epoch)+','+str((test_acc*100)/TEST_SIZE)+'\n')
				test_loss_log.write(str(epoch)+','+str(test_loss)+'\n')
			
			X, Y = data_handler.get_train_batch()
			X = SCALER.transform(X)

			_y = self.forward_pass(np.array(X).T)

			train_loss = 0.0

			y = np.zeros(_y.shape)
			for i in range(BATCH_SIZE) :
				y[Y[i]][i] = 1.0
				train_loss += -np.log(_y[Y[i],i])

			train_loss /= BATCH_SIZE

			train_loss_log.write(str(epoch)+','+str(train_loss)+'\n')

			delta[self.nl] = _y - y

			for l in range(self.nl-1,1,-1) :
				delta[l] = np.dot(self.W[l].T, delta[l+1])*self.activation[l-2](self.z[l], derivative=True)

			for l in range(1,self.nl) :
				vW[l] = (gamma*vW[l]) + (alpha * (((1.0/BATCH_SIZE)*np.dot(delta[l+1], self.a[l].T)) + lamda*self.W[l]) )
				self.W[l] = self.W[l] - vW[l]

				vb[l] = (gamma*vb[l]) + (alpha * (np.sum(delta[l+1], axis=1))).reshape(self.b[l].shape)
				self.b[l] = self.b[l] - vb[l]

			print(clear, end="")

		config_log.close()
		train_loss_log.close()
		test_loss_log.close()
		test_acc_log.close()
		print ("\nTraining Completed!")
		print ('-------------------------------------------------')
		# self.fixed_test(data_handler, name=name)

	def fixed_test(self, data_handler, name='NN_1') :

		SCALER = data_handler.scaler;
		X, Y = data_handler.get_fixed_test()
		X_ = SCALER.transform(X)

		TEST_BATCH_SIZE = data_handler.TEST_BATCH_SIZE

		_y = self.forward_pass(np.array(X_).T)

		for i in range(TEST_BATCH_SIZE) :
			display_img(X[i], 28, 28, "log/"+name+"/random_test/"+str(i+1)+".png")
			with open("log/"+name+"/random_test/"+str(i+1)+".csv", "w") as f :
				f.write('class, probability\n')
				for c in range(len(_y[:,i])) :
					f.write(str(c)+', '+str(_y[c,i])+'\n')


	def random_test(self, data_handler, name='NN_1') :

		SCALER = data_handler.scaler;
		X, Y = data_handler.get_test_batch()
		X_ = SCALER.transform(X)

		TEST_BATCH_SIZE = data_handler.TEST_BATCH_SIZE

		_y = self.forward_pass(np.array(X_).T)

		for i in range(TEST_BATCH_SIZE) :
			display_img(X[i], 28, 28, "log/"+name+"/random_test/"+str(i+1)+".png")
			with open("log/"+name+"/random_test/"+str(i+1)+".csv", "w") as f :
				f.write('class, probability\n')
				for c in range(len(_y[:,i])) :
					f.write(str(c)+', '+str(_y[c,i])+'\n')

