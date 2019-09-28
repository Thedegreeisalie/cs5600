#!/usr/bin/python

import cs5600_6600_f19_hw04
import numpy as np
import network2
from mnist_loader import load_data_wrapper

train_d, valid_d, test_d = load_data_wrapper()

#net = network2.Network([784,30,10], cost=network2.CrossEntropyCost)
#
#net_stats = net.SGD(train_d, 2, 10, 0.5, lmbda = 5.0, evaluation_data = valid_d, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
#
#evaluation_costs, evaluation_accuracies, training_costs, training_accuracies = net_stats
#
#num_epochs = np.arange(0, len(evaluation_costs))

d = cs5600_6600_f19_hw04.collect_1_hidden_layer_net_stats(10, 11, network2.CrossEntropyCost, 2, 10, 0.1, 0.0,train_d, test_d)
print(d)
