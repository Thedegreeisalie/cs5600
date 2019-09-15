
#########################################
# module: hw02_f19.py
# author: Jer Moore a02082167 
#########################################

import numpy as np
import math
import pickle
from hw02_f19_data import *

# save() function to save the trained network to a file
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)

# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn

def build_nn_wmats(mat_dims):
    # will be given a tuple of the dimensions of each layer must generate the weights for the distances between them 
    np.random.seed(1);
    weight_matrices = []
    index = 0;
    while index < len(mat_dims)-1: 
        weight_matrices.append(np.random.randn(mat_dims[index],mat_dims[index+1])) 
        index+=1
    return weight_matrices

#sigmoid from https://gist.github.com/jovianlin/805189d4b19332f8b8a79bbd07e3f598
def sigmoid(x, derivative=False):
      return x*(1-x) if derivative else 1/(1+np.exp(-x))

#this is the routine used for the 3 layer and, or and xor ANNs
def build_231_nn():
    return build_nn_wmats((2,3,1))

def build_221_nn():
    return build_nn_wmats((2,2,1))

#this is the routine used for the 4 layer and, or and xor ANNs
def build_2331_nn():
    return build_nn_wmats((2,3,3,1))

# the routines that were used to build the not ANN
def build_121_nn():
    return build_nn_wmats((1,2,1))

def build_1331_nn():
    return build_nn_wmats((1,3,3,1))

#the routine that was used for the boolean expression ANN
def build_4o1_nn():
    return build_nn_wmats((4,16,1))
def build_4o31_nn():
    return build_nn_wmats((4,16,3,1))

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    W1, W2 = build()
    for j in range(numIters):
        #feeding forward
        Z2 = np.dot(X,W1)
        a2 = sigmoid(Z2)
        Z3 = np.dot(a2,W2)
        yHat = sigmoid(Z3)
        #backPropigation
        yHat_error = y-yHat
        yHat_delta = yHat_error * sigmoid(yHat, derivative=True)
        a2_error = yHat_delta.dot(W2.T)
        a2_delta = a2_error * sigmoid(a2, derivative=True)
        W2 += a2.T.dot(yHat_delta)
        W1 += X.T.dot(a2_delta)
    return W1, W2

def train_4_layer_nn(numIters, X, y, build):
    W1, W2, W3 = build()
    for j in range(numIters):
        #feeding forward
        Z2 = np.dot(X,W1)
        a2 = sigmoid(Z2)
        Z3 = np.dot(a2,W2)
        a3 = sigmoid(Z3)
        Z4 = np.dot(a3,W3)
        yHat = sigmoid(Z4)
        #backPropigation
        yHat_error = y-yHat
        yHat_delta = yHat_error * sigmoid(yHat, derivative=True)
        a3_error = yHat_delta.dot(W3.T)
        a3_delta = a3_error * sigmoid(a3, derivative=True)
        a2_error = a3_delta.dot(W2.T)
        a2_delta = a2_error * sigmoid(a2, derivative=True)
        W3 += a3.T.dot(yHat_delta)
        W2 += a2.T.dot(a3_delta)
        W1 += X.T.dot(a2_delta)
    return W1, W2, W3

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2 = wmats
    Z2 = np.dot(x,W1)
    a2 = sigmoid(Z2)
    Z3 = np.dot(a2, W2)
    yHat = sigmoid(Z3)
    if (thresh_flag):
        if(thresh < yHat):
            return 1
        else:
            return 0
    return yHat 

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    # your code here
    W1, W2, W3 = wmats
    Z2 = np.dot(x,W1)
    a2 = sigmoid(Z2)
    Z3 = np.dot(a2, W2)
    a3 = sigmoid(Z3)
    Z4 = np.dot(a3, W3)
    yHat = sigmoid(Z4)
    if (thresh_flag):
        if(thresh < yHat):
            return 1
        else:
            return 0
    return yHat 
    pass

## Remember to state in your comments the structure of each of your
## ANNs (e.g., 2 x 3 x 1 or 2 x 4 x 4 x 1) and how many iterations
## it took you to train it.
        
     




    

    
