import tensorflow as tf
import numpy as np

#hidden_layers defines the number of layers +2 you will have
#nodes_per_layer is an array of size hidden_layers + 2 with integers
#including input and output
#data is the data i guess

# search the mistake

#from me may be (it is) wrong
def nn_model(number_layers, nodes_per_layer, data):
    npl = nodes_per_layer
    current_layer = tf.Variable(data)

    for i in range(number_layers-2):
        hidden_layer = {'weights':tf.Variable(tf.random_normal([npl[i], npl[i+1]])),
                        'biases':tf.Variable(tf.random_normal([npl[i+1]]))}
        current_layer = tf.add(tf.matmul(current_layer, hidden_layer['weights']), hidden_layer['biases'])
        current_layer = tf.nn.relu(current_layer)


    output_layer = {'weights':tf.Variable(tf.random_normal([npl[number_layers-2]],[npl[number_layers-1]])),
                    'biases':tf.Variable(tf.random_normal([npl[number_layers-1]]))}

    output_layer = tf.matmul(current_layer, output_layer['weights']) + output_layer['biases']
    return output_layer
