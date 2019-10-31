import tflearn
import glob 
import numpy as np
from tflearn import fully_connected, regression, input_data 
from scipy.io import wavfile

FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
NAME = 'ANN_Buzz1_4Layer.tfl'
SAVE_POINT = f'{FOLDER}' + f'{NAME}'
CHECK_POINT = SAVE_POINT + '.meta'

audio_list = glob.glob('/home/jer/Workspace/cs5600/project1/BUZZ1/**/*.wav/')

#I dunno why bus some of the audio files change the type of the overal array to object I don't want so the if condition here excludes them (assuming the first element is not one of those strange files)
X = np.array([wavfile.read(element)[1] for element in audio_list if np.array([wavfile.read(element)[1], wavfile.read(audio_list[0])[1]]).dtype == np.int16])
#this will get the normalized version of this
X = np.array([audio/float(np.max(audio)) for audio in X])
#finally let's get rid of the startup sounds and finish sounds
X = np.array([cut[int(len(cut)/3):2*int(len(cut)/3)] for cut in X])

def class_search(element):
    if(np.char.find(element, 'bee')[0] > 0):
        return [1,0,0]
    elif(np.char.find(element, 'cricket')[0] > 0):
        return [0,1,0]
    else:
        return [0,0,1]

Y = [class_search(element) for element in audio_list]

input_layer = input_data(shape=[None, 29424])
fc_layer_1  = fully_connected(input_layer, 10024,activation='relu',name='fc_layer_1')
fc_layer_2 = fully_connected(fc_layer_1, 1000,activation='relu',name='fc_layer_2')
fc_layer_3 = fully_connected(fc_layer_2, 100,activation='relu',name='fc_layer_3')
fc_layer_4 = fully_connected(fc_layer_3, 3,activation='relu',name='fc_layer_4')
network = regression(fc_layer_4, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
model = tflearn.DNN(network)

#model.fit(beeX, beeY, validation_set = 0.2, n_epoch=100,shuffle=True,show_metric=True,run_id='ANN_BEE1_3Layer')

#model.save('/home/jer/Workspace/cs5600/project1/trained_nets/ANN_Bee_3Layer.tfl')
#Let's see if there's already a trained network with the right name in FOLDER
#if(os.path.exists(CHECK_POINT)):
#    model.load(SAVE_POINT)
#
model.fit(X, Y, validation_set =0.2, n_epoch=1,shuffle=True,show_metric=True,run_id='ANN_Buzz1_4Layer')
#model.save(SAVE_POINT)

