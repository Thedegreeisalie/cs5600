import tflearn
import os
from tflearn import fully_connected, regression, input_data

FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
NAME = 'ANN_Bee_5Layer.tfl'
SAVE_POINT= f'{FOLDER}' + f'{NAME}'
CHECK_POINT = SAVE_POINT + '.meta'

beeX, beeY = tflearn.data_utils.image_preloader('BEE1/class_labels.txt', image_shape=(32,32), mode='file', categorical_labels=True, normalize=True)

input_layer = input_data(shape=[None, 32, 32, 3])
fc_layer_1  = fully_connected(input_layer, 1024,activation='relu',name='fc_layer_1')
fc_layer_2 = fully_connected(fc_layer_1, 512,activation='relu',name='fc_layer_2')
fc_layer_3 = fully_connected(fc_layer_2, 104,activation='relu',name='fc_layer_3')
fc_layer_4 = fully_connected(fc_layer_3, 10,activation='softmax',name='fc_layer_4')
fc_layer_5 = fully_connected(fc_layer_4, 2,activation='softmax',name='fc_layer_5')
network = regression(fc_layer_5, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
model = tflearn.DNN(network)

if(os.path.exists(CHECK_POINT)):
    model.load(SAVE_POINT)

model.fit(beeX, beeY, validation_set = 0.2, n_epoch=100,shuffle=True,show_metric=True,run_id='ANN_BEE1_5Layer')
model.save(SAVE_POINT)

