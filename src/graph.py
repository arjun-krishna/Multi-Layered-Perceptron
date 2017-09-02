"""
Code to visualize the Training Progress plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models1 = ['NN_3', 'NN_1', 'NN_2', 'NN_7']
models2 = ['NN_6', 'NN_4', 'NN_5', 'NN_8']

models = models2

labels = ['0.001', '0.01', '0.05', 'sched (init = 0.02, decay=0.85)']

handles = []
for i in range(len(models)) :

	model = models[i]
	test_acc = pd.read_csv('log/'+model+'/test_acc.csv')
	test_loss = pd.read_csv('log/'+model+'/test_loss.csv')
	train_loss = pd.read_csv('log/'+model+'/train_loss.csv')

	handle, = plt.plot(test_acc['epoch'], test_acc[' test_accuracy'], label='alpha = '+labels[i])
	handles.append(handle)

plt.legend(handles=handles)
plt.xlabel('epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs epoch')
plt.show()

handles = []

for i in range(len(models)) :

	model = models[i]
	test_acc = pd.read_csv('log/'+model+'/test_acc.csv')
	test_loss = pd.read_csv('log/'+model+'/test_loss.csv')
	train_loss = pd.read_csv('log/'+model+'/train_loss.csv')

	handle, = plt.plot(test_loss['epoch'], test_loss[' test_loss'],label='alpha = '+labels[i])
	handles.append(handle)

plt.legend(handles=handles)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Test Loss vs epoch')
plt.show()


for i in range(len(models)) :

	model = models[i]

	test_acc = pd.read_csv('log/'+model+'/test_acc.csv')
	test_loss = pd.read_csv('log/'+model+'/test_loss.csv')
	train_loss = pd.read_csv('log/'+model+'/train_loss.csv')

	train, = plt.plot(train_loss['epoch'], train_loss[' train_loss'],label='train loss')
	test, = plt.plot(test_loss['epoch'], test_loss[' test_loss'],label='test loss')

	plt.legend(handles=[train, test])
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.ylim(0,2.5)
	plt.title('Cross Entropy Loss vs epoch [alpha = '+labels[i]+']')
	plt.show()

