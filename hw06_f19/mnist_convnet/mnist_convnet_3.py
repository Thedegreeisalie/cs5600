#!/usr/bin/python

################################################
# module: mnist_convnet_3.py
# bugs to vladimir dot kulyukin at usu dot edu
################################################

# uncomment if in Py2.
from __future__ import division, print_function

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle
import tflearn.datasets.mnist as mnist

# X is the training data set; Y is the labels for X.
# testX is the testing data set; testY is the labels for testX.
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y = shuffle(X, Y)
testX, testY = shuffle(testX, testY)
trainX = X[0:50000]
trainY = Y[0:50000]
validX = X[50000:]
validY = Y[50000:]
# make sure you reshape the training and testing
# data as follows.
trainX = trainX.reshape([-1, 28, 28, 1])
testX  = testX.reshape([-1, 28, 28, 1])

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# change these two paths accordingly.
VALID_X_PATH = '/home/jer/Workspace/cs5600/hw06_f19/mnist_convnet/valid_x.pck'
VALID_Y_PATH = '/home/jer/Workspace/cs5600/hw06_f19/mnist_convnet/valid_y.pck'
save(validX, VALID_X_PATH)
save(validY, VALID_Y_PATH)

def build_tflearn_convnet_3():
    # we have a 28x28 input layer.
    input_layer = input_data(shape=[None, 28, 28, 1])
    # of 20x24x24 feature maps; 
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    # the convolutional layer of 20 24x24 features is connected
    # to a max pooling layer with a pool window of 2x2; this
    # gives us 20 12x12 features.
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    # convolutional layer of 40 22x22
    conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=3,
                       activation='relu',
                       name='conv_layer_2')
    # pooling layer of 40 11x11 features
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    # convolutional layer of 80 10x10 features
    conv_layer_3 = conv_2d(pool_layer_2,
                       nb_filter=80,
                       filter_size=2,
                       activation='relu',
                       name='conv_layer_3')
    # pooling layer of 40 11x11 features
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    # fully connected layer of 100 sigmoid neurons.
    fc_layer_1  = fully_connected(pool_layer_3, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    # fully connected layer of 10 neurons; the activation
    # function for this layer is softmax.
    fc_layer_2 = fully_connected(fc_layer_1, 10,
                                 activation='softmax',
                                 name='fc_layer_2')
    # finally, the network is trained with SGD, the loss
    # function is categorical_crossentropy, which is a standard
    # option, the learning rate of the network is 0.1.
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    # the network is turned into a model.
    model = tflearn.DNN(network)
    return model
    # your code here

# the model is trained for a specified NUM_EPOCHS
# these are the numbers I used to train my tflearn
# nets for this experiment. Feel free to change them.  
NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = build_tflearn_convnet_3()
MODEL.fit(trainX, trainY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(testX, testY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='MNIST_ConvNet_3')

# let me raise my text size at you: DO NOT FORGET TO PERSIST
# YOUR TRAINED MODELS. THIS IS AN IMPORTANT COMMAND.
SAVE_CONVNET_PATH = '/home/jer/Workspace/cs5600/hw06_f19/hw06_my_net_03.tfl'
MODEL.save(SAVE_CONVNET_PATH)

# this is just to make sure that you've trained everything
# correctly.
# print(model.predict(testX[0].reshape([-1, 28, 28, 1])))
