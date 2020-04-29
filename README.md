# MNIST_tensorflow
This repository is a Tensorflow implementation of MNIST course. Is to solve the handwritten digit recognition problem. The project is used Convolutional neural network (CNN) as the model, the model input is a 1x784 vector (a 28x28 MNIST handwritten digit image is flattened to a 1x784 vector), the model output is a 1x10 vector, is the predicted distribution. 

# Dependencies
Python 3.6
Tensorflow 1.8.0
Numpy
Opencv 3.4.2

# Train: 
python Train_MNIST.py
# Test: 
python Test_MNIST.py
# Covert .ckpt model to .pb model: 
python Ckpt_to_pb_MNIST.py
# Covert MNIST data 1x784 vector to 28x28 image and save as .jpg: 
python Covert_MNIST_image.py
# Define CNN neural network: 
Network.py
