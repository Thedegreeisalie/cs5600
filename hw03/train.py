#!/usr/bin/python


import random
import os
import pickle
import numpy as np
from mnist_loader import load_data_wrapper
from network import Network


train_d, valid_d, test_d = load_data_wrapper()


def train_nets(networks, eta_vals, mini_batch_sizes, num_epochs, path):
    i = 1;
    for net in networks:
        i+=1
        batch_size = random.choice(mini_batch_sizes)
        eta = random.choice(eta_vals)
        net[1].SGD(train_d, num_epochs, random.choice(mini_batch_sizes), random.choice(eta_vals), test_data=test_d)
        save(net[1], path + net[0] + "_" + batch_size + "_" +  str(int(eta*100)) + "_" + ".pck")

def evaluate_net_ensemble(net_ensemble, test_data):
    #need to classify the value of an image based on what the majority of the nets classify it as
    results = []
    for (x,y) in test_data: 
        output = [(np.argmax(net[1].feedforward(x)), y) for net in net_ensemble]
        results.append(most_common(output))
    return (sum(int( x == y) for (x,y) in results), len(test_data))


# thanks to stackoverflow fof helping with this
# https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list#1518632
def most_common(lst):
        return max(set(lst), key=lst.count)

# save() function to save the trained network to a file
# Written by professor k
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)

# restore() function to restore the file
# Written by professor k
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn

def load_nets(folder_path):
    ensemble = []
    for file_name in os.listdir(folder_path):
        ensemble.append((folder_path + file_name, load(folder_path + file_name)))
    return ensemble


#ensemble_dir = "/home/jer/Workspace/cs5600/hw03/ensemble/"
#ensemble = load_nets(ensemble_dir)
#eta_vals = (0.1, 0.25, 0.3, 0.4, 0.5)
#mini_batch_sizes = (5,10,15,20)

#train_nets(ensemble, eta_vals, mini_batch_sizes, 1, ensemble_dir )
#evaluation = evaluate_net_ensemble(ensemble, valid_d)

