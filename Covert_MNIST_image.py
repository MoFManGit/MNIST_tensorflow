"""
Used to covert MNIST image data set to 28x28 image.
File author: LYJ
Date: Jan 2020
"""
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import matplotlib.pyplot as plt
import cv2


######  Default parameters  ######
MNIST_DIR           = './MNIST_data'
COVERT_IMAGE_INDEX  = 1
IMAGE_SAVE_PATH     = './MNIST_image/MNIST_1.jpg'



######  Add command line arguments  ######
def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(description='Test the pretrained model for MNIST.')
    parser.add_argument('--MNIST_dir',
                        help = 'Directory of MNIST testing data.',
                        default = MNIST_DIR)
    parser.add_argument('--covert_image_index',
                        help = """covert image index of MNIST.""",
                        default = COVERT_IMAGE_INDEX,
                        type=int)
    parser.add_argument('--image_save_path',
                        default = IMAGE_SAVE_PATH,
                        help = 'Save path of the coverted image.')

    return parser



if (__name__ == '__main__'):

    ###### Get command line arguments and parameter assignment ######
    args = setup_parser().parse_args()

    MNIST_dir          = args.MNIST_dir
    covert_image_index = args.covert_image_index
    image_save_path    = args.image_save_path


    ###### Get the test data ######
    mnist = input_data.read_data_sets(MNIST_dir, one_hot = True)

    print('mnist.test.images.shape', mnist.test.images.shape)
    print('mnist.test.labels.shape', mnist.test.labels.shape)

    # show the MNIST iamge
    image_vector = mnist.test.images[covert_image_index,:] 
    image_vector = image_vector.reshape(1, 784) 
    image = image_vector.reshape(28, 28) 

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print('image.shape', image.shape)
    #plt.figure() 
    #plt.imshow(image) 
    #plt.show()  # close the image window to continue
    
    cv2.imwrite(image_save_path, 255.0 * image)






