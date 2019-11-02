import tflearn
import glob 
import os
import numpy as np
from tflearn import fully_connected, regression, input_data , dropout
from scipy.io import wavfile

FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
NAME = 'ANN_BUZZ1_dropout.tfl'
SAVE_POINT = f'{FOLDER}' + f'{NAME}'
CHECK_POINT = SAVE_POINT + '.meta'

def class_search(element):
    if(np.char.find(element, 'cricket') > 0 ):
        return np.array([0,1,0], dtype = np.int16) 
    elif(np.char.find(element, 'bee') > 0):
        return np.array([1,0,0], dtype=np.int16) 
    else:
        return np.array([0,0,1], dtype = np.int16) 

def load_BUZZ1():
    cricket_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/cricket/*.wav')
    bee_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/bee/*.wav')
    noise_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/noise/*.wav')
    #my computer can't handle all the data at one time so I'm going to randomly sample 6000 of the sequences but I want to sample them evenly from each group
    test_bee_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/out_of_sample_data_for_validation/bee_test/*.wav')
    test_cricket_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/out_of_sample_data_for_validation/cricket_test/*.wav')
    test_noise_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/out_of_sample_data_for_validation/noise_test/*.wav')
    cricket_list = [cricket_list[x] for x in np.random.choice(len(cricket_list), 2000)]
    bee_list = [bee_list[x] for x in np.random.choice(len(bee_list), 2000)]
    noise_list = [noise_list[x] for x in np.random.choice(len(noise_list), 2000)]
    audio_list = cricket_list + bee_list + noise_list
    test_audio_list = test_cricket_list + test_bee_list + test_noise_list
    obj_list = np.array([(wavfile.read(element)[1], class_search(element)) for element in audio_list])
    test_list = np.array([(wavfile.read(element)[1], class_search(element)) for element in test_audio_list])
    X = np.array([elem[0] for elem in obj_list])
    Y = np.array([elem[1] for elem in obj_list])
    testX = np.array([elem[0] for elem in test_list])
    testY = np.array([elem[1] for elem in test_list])
    X = np.array([elem[int((len(elem) - 44100)/2):int((len(elem) - 44100)/2 + 44100)] for elem in X])
    X = np.array([audio / float(np.max(audio)) for audio in X], dtype=np.float16)
    testX = np.array([elem[int((len(elem) - 44100)/2):int((len(elem) - 44100)/2 + 44100)] for elem in testX])
    testX = np.array([audio / float(np.max(audio)) for audio in testX], dtype=np.float16)
    return X, Y, testX, testY

X, Y, testX, testY = load_BUZZ1()

X = X.reshape([-1, 441, 100, 1])
testX = testX.reshape([-1, 441, 100, 1])

input_layer = input_data(shape=[None, 441, 100, 1])

fc_layer_1  = fully_connected(input_layer, 6650 , activation='tanh',name='fc_layer_1')
dropout_1 = dropout(fc_layer_1, 0.8)

fc_layer_2 = fully_connected(dropout_1, 665,activation='tanh',name='fc_layer_2')
dropout_2 = dropout(fc_layer_2, 0.8)

fc_layer_3 = fully_connected(dropout_2, 100,activation='relu',name='fc_layer_3')
dropout_3 = dropout(fc_layer_3, 0.5)

fc_layer_4 = fully_connected(dropout_3, 10,activation='softmax',name='fc_layer_4')

fc_layer_5 = fully_connected(fc_layer_4, 3,activation='softmax',name='fc_layer_5')

network = regression(fc_layer_5, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
model = tflearn.DNN(network)

#model.fit(beeX, beeY, validation_set = 0.2, n_epoch=100,shuffle=True,show_metric=True,run_id='ANN_BEE1_3Layer')

#model.save('/home/jer/Workspace/cs5600/project1/trained_nets/ANN_Bee_3Layer.tfl')
#Let's see if there's already a trained network with the right name in FOLDER
#if(os.path.exists(CHECK_POINT)):
#    model.load(SAVE_POINT)

model.fit(X, Y, validation_set=(testX, testY), n_epoch=10, shuffle=True, show_metric=True, run_id=f'{NAME}')
model.save(SAVE_POINT)

