import tflearn
import os
from tflearn import fully_connected, regression, input_data, conv_2d, max_pool_2d

FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
NAME = 'CNN_Bee_4Layer.tfl'
SAVE_POINT= f'{FOLDER}' + f'{NAME}'
CHECK_POINT = SAVE_POINT + '.meta'

beeX, beeY = tflearn.data_utils.image_preloader('BEE1/class_labels.txt', image_shape=(32,32), mode='file', categorical_labels=True, normalize=True)

input_layer = input_data(shape=[None, 32, 32, 3])
conv_layer = conv_2d(input_layer, nb_filter = 20, filter_size=10, activation='relu', name='conv_layer1')
pool_layer = max_pool_2d(conv_layer, 3, name='pool_layer_1')
fc_layer_1  = fully_connected(pool_layer, 100,activation='relu',name='fc_layer_1')
fc_layer_2 = fully_connected(fc_layer_1, 2,activation='softmax',name='fc_layer_2')
network = regression(fc_layer_2, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
model = tflearn.DNN(network)

#model.fit(beeX, beeY, validation_set = 0.2, n_epoch=100,shuffle=True,show_metric=True,run_id='ANN_BEE1_3Layer')

#model.save('/home/jer/Workspace/cs5600/project1/trained_nets/ANN_Bee_3Layer.tfl')
#Let's see if there's already a trained network with the right name in FOLDER
if(os.path.exists(CHECK_POINT)):
    model.load(SAVE_POINT)

model.fit(beeX, beeY, validation_set = 0.2, n_epoch=200,shuffle=True,show_metric=True,run_id='ANN_BEE1_3Layer')
model.save(SAVE_POINT)

