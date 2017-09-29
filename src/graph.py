"""
Code to visualize the Training Progress plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


models = ['model_1']

labels = ['sched (init = 0.02, decay=0.85)']

handles = []
for i in range(len(models)) :

	model = models[i]
	test_acc = pd.read_csv('log/'+model+'/test_acc.csv')
	test_loss = pd.read_csv('log/'+model+'/test_loss.csv')
	train_loss = pd.read_csv('log/'+model+'/train_loss.csv')

	handle, = plt.plot(test_acc['iteration'], test_acc[' test_accuracy'], label='alpha = '+labels[i])
	handles.append(handle)

plt.legend(handles=handles)
plt.xlabel('iteration')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs iterations')
plt.show()

handles = []

for i in range(len(models)) :

	model = models[i]
	test_acc = pd.read_csv('log/'+model+'/test_acc.csv')
	test_loss = pd.read_csv('log/'+model+'/test_loss.csv')
	train_loss = pd.read_csv('log/'+model+'/train_loss.csv')

	handle, = plt.plot(test_loss['iteration'], test_loss[' test_loss'],label='alpha = '+labels[i])
	handles.append(handle)

plt.legend(handles=handles)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Test Loss vs iterations')
plt.show()


for i in range(len(models)) :

	model = models[i]

	test_acc = pd.read_csv('log/'+model+'/test_acc.csv')
	test_loss = pd.read_csv('log/'+model+'/test_loss.csv')
	train_loss = pd.read_csv('log/'+model+'/train_loss.csv')

	train, = plt.plot(train_loss['iteration'], train_loss[' train_loss'],label='train loss')
	test, = plt.plot(test_loss['iteration'], test_loss[' test_loss'],label='test loss')

	plt.legend(handles=[train, test])
	plt.xlabel('iteration')
	plt.ylabel('Loss')
	plt.ylim(0,2.5)
	plt.title('Cross Entropy Loss vs iterations [alpha = '+labels[i]+']')
	plt.show()

