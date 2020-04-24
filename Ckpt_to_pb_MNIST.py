"""
Used to train a image classifer model to MNIST image data set.
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

from tensorflow.python.framework.graph_util import convert_variables_to_constants


######  Default parameters  ######
MNIST_DIR           = './MNIST_data'
MODEL_PATH          = './models/ckpt/MNIST_50_0.01_500_final.ckpt'
TEST_IMAGE_INDEX    = 500
OUTPUT_PB_PATH      = './models/pb/MNIST.pb'



######  Add command line arguments  ######
def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(
                description='Test the pretrained model for MNIST.')
    parser.add_argument('--MNIST_dir',
                        help='Directory of MNIST testing data.',
                        default=MNIST_DIR)
    parser.add_argument('--model_path',
                        default=MODEL_PATH,
                        help='Pretrained model path.')
    parser.add_argument('--test_image_index',
                        help="""test image index of MNIST.""",
                        default=TEST_IMAGE_INDEX,
                        type=int)
    parser.add_argument('--output_pb_path',
                        help='Desired output pb file path.',
                        default=OUTPUT_PB_PATH)


    return parser



if __name__ == '__main__':

    ###### Get command line arguments and parameter assignment ######
    args = setup_parser().parse_args()

    MNIST_dir        = args.MNIST_dir
    model_path       = args.model_path
    test_image_index = args.test_image_index
    output_pb_path   = args.output_pb_path


    ###### Get the test data ######
    mnist = input_data.read_data_sets(MNIST_dir,one_hot=True)

    print('mnist.test.images.shape', mnist.test.images.shape)
    print('mnist.test.labels.shape', mnist.test.labels.shape)

    # show the MNIST iamge
    image_vector = mnist.test.images[test_image_index,:] 
    image_vector = image_vector.reshape(1,784) 
    image = image_vector.reshape(28,28) 
    plt.figure() 
    plt.imshow(image) 
    plt.show()  # close the image window to continue
    
    ###### Create the neure net-work graph ######
    print( 'Resetting default graph.' )
    # get clear of the default graph for the tendorflow
    tf.reset_default_graph()                                                                            
    with tf.variable_scope('MNIST_class_Network'):
        X = tf.placeholder(tf.float32, [None,784]) 
        Y = create_net( X )

    ###### Define the model saver  ######
    # model_saver used to restore the model to the session.
    model_saver = tf.train.Saver()
        
    with tf.Session() as sess:
        # load the pretrained mode *.ckpt for sess as specified by model_path, this load the weight, not the net define information
        model_saver.restore( sess, model_path )          
        start_time = time.clock()

        # pass the input MNIST image (a vector) to the net and get the predicted result.
        predict = sess.run( Y, feed_dict = { X:image_vector } )      
        predict = tf.squeeze(predict)

        print('True labe is   :', sess.run( tf.argmax( mnist.test.labels[test_image_index,:]) ) ) 
        print('Predict labe is:', sess.run( tf.argmax( predict) ) ) 
 
        stop_time = time.clock()
        print("Time consumed: %f" % (stop_time-start_time)) 

        
        graph = convert_variables_to_constants(sess, sess.graph_def, ['MNIST_class_Network/output'])
        
        with tf.gfile.GFile( output_pb_path, "wb") as f:  
            f.write(graph.SerializeToString()) 
        print("%d ops in the final graph." % len(graph.node))
        
        tf.summary.FileWriter('./models/pb/' , tf.get_default_graph())
         
        










