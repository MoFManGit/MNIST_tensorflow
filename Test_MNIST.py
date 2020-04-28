"""
Used to test the pre-trained model for MNIST 28x28 image.
File author: LYJ
Date: Jan 2020
"""
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Network import create_net
import argparse
import os
import time
import cv2


######  Default parameters  ######
MNIST_IMAGE_PATH    = './MNIST_image/MNIST_1.jpg'
MODEL_PATH          = './models/ckpt/MNIST_50_0.01_1000_final.ckpt'


######  Add command line arguments  ######
def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(description = 'Test the pretrained model for MNIST image.')
    parser.add_argument('--MNIST_image_path',
                        help = 'The path of the MNIST image.',
                        default = MNIST_IMAGE_PATH)
    parser.add_argument('--model_path',
                        default = MODEL_PATH,
                        help = 'Pretrained model path.')

    return parser



if (__name__ == '__main__'):

    ###### Get command line arguments and parameter assignment ######
    args = setup_parser().parse_args()

    MNIST_image_path = args.MNIST_image_path
    model_path       = args.model_path

    ###### Get the test data ######
    image = cv2.imread(MNIST_image_path, cv2.IMREAD_GRAYSCALE)
    print('image.shape', image.shape)
    image_vector = image.reshape(1, 784) 

    # show the MNIST iamge
    plt.figure() 
    plt.imshow(image) 
    plt.show()  # close the image window to continue
    
    ###### Create the neure net-work graph ######
    print( 'Resetting default graph.' )
    # get clear of the default graph for the tendorflow
    tf.reset_default_graph()                                                                            
    with tf.variable_scope('MNIST_class_Network'):
        X = tf.placeholder(tf.float32, [None,784]) 
        Y = create_net(X)

    ###### Define the model saver  ######
    # model_saver used to restore the model to the session.
    model_saver = tf.train.Saver()
        
    with tf.Session() as sess:
        # load the pretrained mode *.ckpt for sess as specified by model_path, this load the weight, not the net define information
        model_saver.restore(sess, model_path)          
        start_time = time.clock()

        # pass the input MNIST image (a vector) to the net and get the predicted result.
        predict = sess.run(Y, feed_dict = {X : image_vector})      
        predict = tf.squeeze(predict)

        print('Predict labe is:', sess.run(tf.argmax(predict))) 
 
        stop_time = time.clock()
        print("Time consumed: %f" % (stop_time - start_time))  
        










