# MNIST_tensorflow
This repository is a Tensorflow implementation of MNIST course. Is to solve the handwritten digit recognition problem. The project is used Convolutional neural network (CNN) as the model, the model input is a 1x784 vector (a 28x28 MNIST handwritten digit image is flattened to a 1x784 vector), the model output is a 1x10 vector, is the predicted distribution.<br>
Overview of MNIST handwritten digit images<br>
![image](https://github.com/MoFManGit/MNIST_tensorflow/blob/master/README_images/mnistdataset.jpg)

# Dependencies
Python 3.6<br>
Tensorflow 1.8.0<br>
Numpy<br>
Opencv 3.4.2<br>

# Usage
## Train:
Train_MNIST.py is the training code file, ./MNIST_data is stored the MNIST training dataset. ./models/ckpt/ is stored the pretrained model .ckpt file, ./summaries/ stored the training log of loss curve. you can use the default parameter to train the model by run:<br>
`python Train_MNIST.py`

## Covert MNIST data to image: 
run: `python Covert_MNIST_image.py` to covert a MNIST data 1x784 vector to 28x28 image and save as .jpg

## Test: 
run: `python Test_MNIST.py` is to used the pretrained model ./models/ckpt/MNIST_50_0.01_1000_final.ckpt for a 28x28 MNIST image  inference process

## Covert .ckpt model to .pb model: 
run: `python Ckpt_to_pb_MNIST.py` to covert a pretrained model of .ckpt file to .pb file.

## CNN architecture: 
Network.py is defined the architecture of CNN, the model is composed by a convolutional layer, a max pooling layer and two fully connected layer.







