#!/usr/bin/python

####################################################
# CS 5600/6600/7890: Assignment 1: Problems 1 & 2
# YOUR NAME Jer Moore
# YOUR A# a02082167
#####################################################

import numpy as np

class and_perceptron:

    def __init__(self):
        # your code here
        self.weight = 0
        self.bias = 0
        pass

    def output(self, x):
        # your code here
        if((x[0] + x[1]) >= 2):
            pass
            return 1
        return 0

class or_perceptron:
    def __init__(self):
        # your code
        self.weight = 0
        self.bias = 0
        pass

    def output(self, x):
        # your code
        if((x[0] + x[1]) >= 1):
            pass
            return 1
        return 0

class not_perceptron:
    def __init__(self):
        # your code
        self.weight = -1 
        self.bias = 0
        pass

    def output(self, x):
        # your code
        if(x + self.weight < 0):
            pass
            return 1
        return 0

class xor_perceptron:
    def __init__(self):
        # your code
        self.weight = 0 
        self.bias = 0
        pass

    def output(self, x):
        # your code
        andp = and_perceptron()
        notp = not_perceptron()
        orp = or_perceptron()
        if(andp.output([notp.output(andp.output(x)), orp.output(x)])):
            pass
            return 1
        return 0
        


class xor_perceptron2:
    def __init__(self):
        # your code
        self.weight = -1 
        self.bias = 0
        pass

    def output(self, x):
        # your code
        if((x[0] + x[1] + self.weight) == 0):
            pass
            return [1]
        return [0]

### ================ Unit Tests ====================

# let's define a few binary input arrays.    
x00 = np.array([0, 0])
x01 = np.array([0, 1])
x10 = np.array([1, 0])
x11 = np.array([1, 1])

# let's test the and perceptron.
def unit_test_01():
    andp = and_perceptron()
    assert andp.output(x00) == 0
    assert andp.output(x01) == 0
    assert andp.output(x10) == 0
    assert andp.output(x11) == 1
    print 'all andp assertions passed...'


# let's test the or perceptron.
def unit_test_02():
    orp = or_perceptron()
    assert orp.output(x00) == 0
    assert orp.output(x01) == 1
    assert orp.output(x10) == 1
    assert orp.output(x11) == 1
    print 'all orp assertions passed...'


# let's test the not perceptron.
def unit_test_03():
    notp = not_perceptron()
    assert notp.output(np.array([0])) == 1
    assert notp.output(np.array([1])) == 0
    print 'all notp assertions passed...'

# let's test the 1st xor perceptron.
def unit_test_04():
    xorp = xor_perceptron()
    assert xorp.output(x00) == 0
    assert xorp.output(x01) == 1
    assert xorp.output(x10) == 1
    assert xorp.output(x11) == 0
    print 'all xorp assertions passed...'

# let's test the 2nd xor perceptron.
def unit_test_05():
    xorp2 = xor_perceptron2()
    assert xorp2.output(x00)[0] == 0
    assert xorp2.output(x01)[0] == 1
    assert xorp2.output(x10)[0] == 1
    assert xorp2.output(x11)[0] == 0
    print 'all xorp2 assertions passed...' 
#unit_test_01()    
#unit_test_02()    
#unit_test_03()    
#unit_test_04()    
#unit_test_05()    
        
        

    
        





