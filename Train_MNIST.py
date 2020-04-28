"""
Used to train a image classifer model to MNIST image data set.
author: LYJ
Date:   April 2020
"""
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Network import create_net
import argparse
import os
import time


######  Default parameters  ######
MNIST_DIR           = './MNIST_data'
NUM_STEPS_BREAK     = 1000
NUM_STEPS_PRINT     = 10
START_LEARN_RATE    = 0.01
BATCH_SIZE          = 50
MODEL_NAME          = 'MNIST'


######  Add command line arguments  ######
def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(description = 'Train a image classifer net to MNIST.')

    parser.add_argument('--MNIST_dir',
                        help = 'Directory of MNIST training data.',
                        default = MNIST_DIR)
    parser.add_argument('--model_name',
                        help = 'Name of model being trained.',
                        default = MODEL_NAME)
    parser.add_argument('--num_steps_break',
                        help = """Max on number of steps.""",
                        default = NUM_STEPS_BREAK,
                        type = int)
    parser.add_argument('--num_steps_print',
                        help = """num steps to print training information.""",
                        default = NUM_STEPS_PRINT,
                        type = int)
    parser.add_argument('--start_learn_rate',
                        help = 'Learning rate for Adam optimizer.',
                        default = START_LEARN_RATE, 
                        type = float)
    parser.add_argument('--batch_size',
                        help = 'Batch size for training.',
                        default = BATCH_SIZE, 
                        type = int)
  
    return parser



if (__name__ == '__main__'):

    ###### Get command line arguments and parameter assignment ######
    args = setup_parser().parse_args()

    MNIST_dir        = args.MNIST_dir
    num_steps_break  = args.num_steps_break
    num_steps_print  = args.num_steps_print
    start_learn_rate = args.start_learn_rate
    batch_size       = args.batch_size
    model_name = args.model_name

    model_name = model_name + '_' + str(batch_size) + '_' + str(start_learn_rate) + '_' + str(num_steps_break)


    ###### Get the training data ######
    mnist = input_data.read_data_sets(MNIST_dir, one_hot = True)

    print('mnist.train.images.shape', mnist.train.images.shape)
    print('mnist.train.labels.shape', mnist.train.labels.shape)

    # show the MNIST iamge
    #image = mnist.train.images[20,:] 
    #image = image.reshape(28,28) 
    #print(mnist.train.labels[20]) 
    #plt.figure() 
    #plt.imshow(image) 
    #plt.show()  # close the image window to continue
    
    ###### Create the neure net-work graph ######
    print( 'Resetting default graph.' )
    # get clear of the default graph for the tendorflow
    tf.reset_default_graph()                                                                            
    with tf.variable_scope('MNIST_class_Network'):
        X = tf.placeholder(tf.float32, [None,784]) 
        Y = create_net(X)

    Y_ = tf.placeholder(tf.float32, [None,10])


    ###### Define the cost function and Training step ######
    # MNIST is a classification problem use cross entropy loss
    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_, logits = Y ))

    # global_step is the variable (0) used to count for iterations, and don't need to be training
    global_step = tf.Variable(0, name = 'global_step', trainable = False) 

    # get the variables of the neure network which is need to be trained                         
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'MNIST_class_Network') 
   
    learning_rate = tf.constant(start_learn_rate)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step, train_vars)

    correct_prediction = tf.equal(tf.argmax(Y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    ###### Define the training log to be saved ######
    with tf.name_scope('summaries'):
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate) 
        tf.summary.scalar('accuracy', accuracy)

    # Dir that we'll later save loss curve
    if (not os.path.exists('./summaries')):  
        os.makedirs('./summaries')

    summary_merged = tf.summary.merge_all()
    full_log_path = './summaries/' + model_name
    train_writer = tf.summary.FileWriter(full_log_path)


    ###### Define the model saver  ######
    # Dir that save final models to
    if not os.path.exists('./models/ckpt'):  
        os.makedirs('./models/ckpt')

    model_saver = tf.train.Saver()
    

    # We must include local variables because of batch pipeline.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    ###### Begin training.  ######
    print('Start training: ',time.asctime(time.localtime(time.time())))
    with tf.Session() as sess:
        # Initialization
        sess.run(init_op)

        current_step = 0
        while (current_step < num_steps_break): 
            batch_x, batch_y = mnist.train.next_batch(batch_size) 
            feed_dict = {X : batch_x, Y_ : batch_y}

            # the current iteration index   
            current_step = sess.run(global_step)                                 

            if (current_step % num_steps_print == 0): 
                train_accuracy = accuracy.eval(feed_dict = feed_dict) 

                # Collect some diagnostic data for Tensorboard.
                summary, loss, learning_rate_val = sess.run([summary_merged,total_loss,learning_rate], feed_dict = feed_dict) 
                train_writer.add_summary(summary, current_step)

                print( 'step:',current_step,'loss = ',loss,'train_accuracy = ',train_accuracy )
                

            train_step.run(feed_dict = feed_dict) 


        # save the final trained model
        model_saver.save(sess, 'models/ckpt/' + model_name + '_final.ckpt')                  
        print('final model saved:' + 'models/ckpt/' + model_name + '_final.ckpt')
        print( 'End of training: ', time.asctime(time.localtime(time.time())))
        print("test accuracy %g " % accuracy.eval(feed_dict = {X : mnist.test.images, Y_ : mnist.test.labels})) 
        
        












