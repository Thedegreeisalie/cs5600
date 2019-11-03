# Each net was trained as follows. 
# Due to my poor note taking I don't know how long some of the BEE1 and BEE2 nets were trained for
# 
#
import tflearn
from tflearn import fully_connected, regression, input_data

FOLDER = '/home/jer/Workspace/cs5600/project1/trained_nets/'
PATH_TO_ANN_BEE1_3LAYER = f'{FOLDER}' + 'ANN_Bee1_3Layer.tfl'
PATH_TO_ANN_BEE1_5LAYER = f'{FOLDER}' + 'ANN_Bee1_5Layer.tfl'
PATH_TO_ANN_BEE2_1S_5LAYER = f'{FOLDER}' + 'ANN_Bee1_5Layer_relu_grayscale.tfl'
PATH_TO_CNN_BEE2_1S_5LAYER = f'{FOLDER}' + 'ANN_Bee1_5Layer_relu_grayscale.tfl'

PATH_TO_ANN_BEE1_4LAYER = f'{FOLDER}' + 'CNN_Bee1_4Layer.tfl'

def ann_bee1_3layer():
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1  = fully_connected(input_layer, 1024,activation='relu',name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 100,activation='softmax',name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,activation='softmax',name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.01)
    model = tflearn.DNN(network)
    return model.load(PATH_TO_ANN_BEE1_3LAYER)
    
ann_bee1_3layer()
