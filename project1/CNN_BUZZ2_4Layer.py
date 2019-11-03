import tflearn
import glob 
import numpy as np
import os
from tflearn import fully_connected, regression, input_data , dropout, conv_2d, max_pool_2d
from scipy.io import wavfile


FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
NAME = 'ANN_Buzz2_5Layer.tfl'
SAVE_POINT = f'{FOLDER}' + f'{NAME}'
CHECK_POINT = SAVE_POINT + '.meta'

def class_search(element):
    if(np.char.find(element, 'cricket') > 0 ):
        return np.array([0,1,0], dtype = np.int16) 
    elif(np.char.find(element, 'bee') > 0):
        return np.array([1,0,0], dtype=np.int16) 
    else:
        return np.array([0,0,1], dtype = np.int16) 

#I dunno why bus some of the audio files change the type of the overal array to object I don't want so the if condition here excludes them (assuming the first element is not one of those strange files)
def load_BUZZ1():
    cricket_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ2/train/cricket_train/*.wav')
    bee_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ2/train/bee_train/*.wav')
    noise_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ2/train/noise_train/*.wav')
    #my computer can't handle all the data at one time so I'm going to randomly sample 6000 of the sequences but I want to sample them evenly from each group
    test_bee_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ2/test/bee_test/*.wav')
    test_cricket_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ2/test/cricket_test/*.wav')
    test_noise_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ2/test/noise_test/*.wav')
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

X = X.reshape(-1, 441, 100, 1)
testX = testX.reshape(-1, 441, 100, 1)

input_layer = input_data(shape=[None, 441, 100, 1])
conv_layer_1  = conv_2d(input_layer,nb_filter=20,filter_size=5,activation='relu',name='conv_layer_1')
pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
conv_layer_2 = conv_2d(pool_layer_1,nb_filter=40,filter_size=3,activation='relu',name='conv_layer_2')
pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
conv_layer_3 = conv_2d(pool_layer_2,nb_filter=80,filter_size=2,activation='relu',name='conv_layer_3')
pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
conv_layer_4 = conv_2d(pool_layer_2,nb_filter=160,filter_size=2,activation='relu',name='conv_layer_4')
pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
fc_layer_1  = fully_connected(pool_layer_3, 100,activation='relu',name='fc_layer_1')
fc_layer_2 = fully_connected(fc_layer_1, 3,activation='softmax',name='fc_layer_2')
network = regression(fc_layer_2, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
model = tflearn.DNN(network)

#model.fit(beeX, beeY, validation_set = 0.2, n_epoch=100,shuffle=True,show_metric=True,run_id='ANN_BEE1_3Layer')

#model.save('/home/jer/Workspace/cs5600/project1/trained_nets/ANN_Bee_3Layer.tfl')
#Let's see if there's already a trained network with the right name in FOLDER
if(os.path.exists(CHECK_POINT)):
    model.load(SAVE_POINT)
#
model.fit(X, Y, validation_set=(testX, testY), n_epoch=50, shuffle=True, show_metric=True, run_id=f'{NAME}')
model.save(SAVE_POINT)

