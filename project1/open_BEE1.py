#!/usr/bin/python3

import cv2
import numpy as np
import glob
import pickle
import tflearn
from skimage import io
from PIL import Image
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
#env variables

PATH_TO_BEE1='/home/jer/Workspace/cs5600/project1/BEE1/'

# takes a string to the image then returns the grayscaled numpy.array
def read_img(path):
    img = io.imread(path)
    
    #img = np.array(Image.fromarray(img).resize((32, 32)))
    return img
    

#takes a path to a folder with a bunch of images returns an array of grayscaled numpy.ndarray 
def read_folder(path):
    file_list = glob.glob(f'{path}/**/*.png') 
    images_as_array = [read_img(item) for item in file_list]
    return images_as_array

def read_BEE1():
    bee_train = read_folder(PATH_TO_BEE1 + 'bee_train') 
    bee_test = read_folder(PATH_TO_BEE1 + 'bee_test') 
    bee_valid = read_folder(PATH_TO_BEE1 + 'bee_valid') 
    no_bee_train = read_folder(PATH_TO_BEE1 + 'no_bee_train') 
    no_bee_test = read_folder(PATH_TO_BEE1 + 'no_bee_test') 
    no_bee_valid = read_folder(PATH_TO_BEE1 + 'no_bee_valid') 
    X = np.append([item for item in bee_train], [item for item in no_bee_train], axis = 0)
    Y = np.append([[0.0,1.0] for item in bee_train], [[1.0,0.0] for item in no_bee_train], axis = 0)
    testX = np.append([item for item in bee_test], [item for item in no_bee_test], axis = 0)
    testY = np.append([[0.0,1.0] for item in bee_test], [[1.0,0.0] for item in no_bee_test], axis = 0)
    validX = np.append([item for item in bee_valid], [item for item in no_bee_valid], axis = 0)
    validY = np.append([[0.0,1.0] for item in bee_valid], [[1.0,0.0]for item in no_bee_valid], axis = 0)
    return X, Y, testX, testY, validX, validY

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# X is the training data set; Y is the labels for X.
# testX is the testing data set; testY is the labels for testX.
X, Y, testX, testY, validX, validY = read_BEE1()
X, Y = shuffle(X, Y)
testX, testY = shuffle(testX, testY)
# make sure you reshape the training and testing
# data as follows.
trainX = X.reshape([-1, 32, 32, 1])
testX  = testX.reshape([-1, 32, 32, 1])

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# change these two paths accordingly.
VALID_X_PATH = '/home/jer/Workspace/cs5600/project1/valid_x.pck'
VALID_Y_PATH = '/home/jer/Workspace/cs5600/project1/valid_y.pck'
save(validX, VALID_X_PATH)
save(validY, VALID_Y_PATH)

def build_tflearn_convnet_4():
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1  = fully_connected(input_layer, 1024,activation='relu',name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 500,activation='softmax',name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 100,activation='softmax',name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 10,activation='softmax',name='fc_layer_4')
    fc_layer_5 = fully_connected(fc_layer_4, 2,activation='softmax',name='fc_layer_5')
    network = regression(fc_layer_5, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
    model = tflearn.DNN(network)
return model

# the model is trained for a specified NUM_EPOCHS
# with a specified batch size; of course, you'll want
# to raise the number of epochs to some larger number.
NUM_EPOCHS = 2 
BATCH_SIZE = 10
MODEL = build_tflearn_convnet_4()
MODEL.fit(trainX, trainY, n_epoch=NUM_EPOCHS,shuffle=True,show_metric=True,run_id='MNIST_ConvNet_proj1')




# let me raise my text size at you: DO NOT FORGET TO PERSIST
# YOUR TRAINED MODELS. THIS IS AN IMPORTANT COMMAND.
SAVE_CONVNET_PATH = '/home/jer/Workspace/cs5600/project1/4_layer.tfl'
MODEL.save(SAVE_CONVNET_PATH)

# this is just to make sure that you've trained everything
# correctly.
# print(model.predict(testX[0].reshape([-1, 28, 28, 1])))

# PATH_TO_BEE1
# Assuming the following file structure
# .
# ├── bee_test
# │   ├── img0
# │   ├── img1
# │   ├── img2
# │   ├── img3
# │   ├── img4
# │   ├── img5
# │   └── img6
# ├── bee_train
# │   ├── img0
# │   ├── img1
# │   ├── img10
# │   ├── img11
# │   ├── img12
# │   ├── img13
# │   ├── img14
# │   ├── img15
# │   ├── img16
# │   ├── img17
# │   ├── img18
# │   ├── img19
# │   ├── img2
# │   ├── img3
# │   ├── img4
# │   ├── img5
# │   ├── img6
# │   ├── img7
# │   ├── img8
# │   └── img9
# ├── bee_valid
# │   ├── img0
# │   └── img1
# ├── no_bee_test
# │   ├── img0
# │   ├── img1
# │   ├── img2
# │   ├── img3
# │   ├── img4
# │   ├── img5
# │   └── img6
# ├── no_bee_train
# │   ├── img0
# │   ├── img1
# │   ├── img10
# │   ├── img11
# │   ├── img12
# │   ├── img13
# │   ├── img14
# │   ├── img15
# │   ├── img16
# │   ├── img17
# │   ├── img18
# │   ├── img19
# │   ├── img2
# │   ├── img3
# │   ├── img4
# │   ├── img5
# │   ├── img6
# │   ├── img7
# │   ├── img8
# │   └── img9
# └── no_bee_valid
#     ├── img0
#     └── img1
# 
# 64 directories, 0 files
