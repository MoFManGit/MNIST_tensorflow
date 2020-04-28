"""
Functions used for the creation of the MNIST image classification network.
author: LYJ
Date:   April 2020
"""

import tensorflow as tf

def create_net(X):

    print('feature shpe: ', X.shape)

    h = tf.reshape(X, [-1,28,28,1])
    print('feature shpe: ', h.shape)
    
    h = conv2d(h, 1, 32, 3, [1,1,1,1])
    print('feature shpe: ', h.shape)

    h = relu(h)
    print('feature shpe: ', h.shape )

    h = max_pool(h, 2, [1,2,2,1])
    print('feature shpe: ', h.shape)

    h = tf.reshape(h, [-1,14*14*32]) 
    print('feature shpe: ', h.shape)

    h = fc_layer(h, 14*14*32, 784, name = 'W1')
    print('feature shpe: ', h.shape)

    h = fc_layer(h, 784, 10, name = 'W2')
    print('feature shpe: ', h.shape)

    h = tf.identity(h, name = 'output')
    #h = tf.reshape(h, h.get_shape(), name = 'output')

    return h

def conv2d(X, n_ch_in, n_ch_out, kernel_size, strides, name = None, padding = 'SAME'):

    """
    Creates the convolutional layer.
    X:              Input tensor
    n_ch_in:        Number of input channels
    n_ch_out:       Number of output channels
    kernel_size:    Dimension of the square-shaped convolutional kernel
    strides:        Length 4 vector of stride information
    name:           Optional name for the weight matrix
    """

    if name is None:
        name = 'W'
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    W = tf.get_variable(name = name,
                        shape = shape,
                        dtype = tf.float32,
                        initializer = tf.random_normal_initializer(stddev=0.1))
    h = tf.nn.conv2d(X,
                     filter = W,
                     strides = strides,
                     padding = padding)
    return h


def relu(X):
    """
    Performs relu on the tensor.
    X:    Input tensor
    """
    return tf.nn.relu(X, name = 'relu')


def max_pool(X, kernel_size, strides, padding = 'SAME'):
    """
    Performs max pooling on the tensor.
    X:              Input tensor
    kernel_size:    Dimension of the square-shaped max pooling kernel
    strides:        Length 4 vector of stride information
    """
    
    ksize = [1, kernel_size, kernel_size, 1]
   
    return tf.nn.max_pool( X, ksize = ksize, strides = strides, padding = padding)


def fc_layer(X, n_neure_in, n_neure_out, name = None, padding = 'SAME'):

    """
    Creates the fully connected layer.
    X:              Input tensor
    n_neure_in:     Number of input neures
    n_neure_out:    Number of output neures
    name:           Optional name for the weight matrix
    """

    if name is None:
        name = 'W'
    shape = [n_neure_in, n_neure_out]
    W = tf.get_variable(name = name,
                        shape = shape,
                        dtype = tf.float32,
                        initializer = tf.random_normal_initializer(stddev = 0.1))
    B = tf.Variable(tf.zeros(n_neure_out))

    h = tf.matmul(X, W) + B 

    return h













