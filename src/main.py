from MLP import *
from DataHandler import *

dh = DataHandler()

network1 = MLP([784,1000,500,250,10],[sigmoid, sigmoid, sigmoid, softmax])
network2 = MLP([784,1000,500,250,10],[sigmoid, sigmoid, sigmoid, softmax])
network3 = MLP([784,1000,500,250,10],[sigmoid, sigmoid, sigmoid, softmax])

network4 = MLP([784,1000,500,250,10],[relu, relu, relu, softmax])
network5 = MLP([784,1000,500,250,10],[relu, relu, relu, softmax])
network6 = MLP([784,1000,500,250,10],[relu, relu, relu, softmax])


# network1.train_mini_batch(dh, name='NN_1', act='sigmoid', alpha=0.01) 
# network2.train_mini_batch(dh, name='NN_2', act='sigmoid', alpha=0.05) 
# network3.train_mini_batch(dh, name='NN_3', act='sigmoid', alpha=0.001)  

# network4.train_mini_batch(dh, name='NN_4', act='relu', alpha=0.01) 
# network5.train_mini_batch(dh, name='NN_5', act='relu', alpha=0.05) 
# network6.train_mini_batch(dh, name='NN_6', act='relu', alpha=0.001)  

# With Scheduling

network1.train_mini_batch(dh, name='model_1', act='sigmoid_sched', alpha=0.02, lrf=0.85)
# network4.train_mini_batch(dh, name='NN_8', act='relu_sched', alpha=0.02, lrf=0.85)

