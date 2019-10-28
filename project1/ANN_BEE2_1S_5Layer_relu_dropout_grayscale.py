import tflearn
import os
from tflearn import fully_connected, regression, input_data, dropout
from tensorflow import image

FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
NAME = 'ANN_Bee2_1S_5Layer_relu_grayscale'
SAVE_POINT= f'{FOLDER}' + f'{NAME}'+ '.tfl'
CHECK_POINT = SAVE_POINT + '.meta'

beeX, beeY = tflearn.data_utils.image_preloader('BEE2_1S/class_labels.txt', image_shape=(90,90), mode='file', categorical_labels=True, normalize=True)

input_layer = input_data(shape=[None, 90, 90, 3])
input_layer = image.rgb_to_grayscale(input_layer)
fc_layer_1  = fully_connected(input_layer, 8100 ,activation='relu',name='fc_layer_1')
dropout_1 = dropout(fc_layer_1, 0.5)
fc_layer_2 = fully_connected(dropout_1, 4000,activation='relu',name='fc_layer_2')
dropout_2 = dropout(fc_layer_2, 0.5)
fc_layer_3 = fully_connected(dropout_2, 500,activation='relu',name='fc_layer_3')
dropout_3 = dropout(fc_layer_3, 0.5)
fc_layer_4 = fully_connected(dropout_3, 100,activation='relu',name='fc_layer_4')
fc_layer_5 = fully_connected(fc_layer_4, 2,activation='softmax',name='fc_layer_5')
network = regression(fc_layer_5, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
model = tflearn.DNN(network)

#model.fit(beeX, beeY, validation_set = 0.2, n_epoch=100,shuffle=True,show_metric=True,run_id='ANN_BEE1_3Layer')

#model.save('/home/jer/Workspace/cs5600/project1/trained_nets/ANN_Bee_3Layer.tfl')
#Let's see if there's already a trained network with the right name in FOLDER
if(os.path.exists(CHECK_POINT)):
    model.load(SAVE_POINT)

model.fit(beeX, beeY, validation_set = 0.2, n_epoch=25,shuffle=True,show_metric=True,run_id=f'{NAME}')
model.save(SAVE_POINT)

