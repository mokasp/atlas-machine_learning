# Transfer Learning

Transfer learning is the process of re-purposing an already trained machine learning model to classify a different set of data than the one it was originally trained on.

This is useful in the domain of deep-learning because training a model from scratch can be very resource intenseive and may take quite a bit of time. In addition, for many tasks you may want to perform you likely do not have a large enough dataset to train the model well enough.

(feature extraction)

(fine-tuning)

By taking a pre-trained model, and removing the top layers (the layers responsible for classifying the data), you are left with all the feature extraction layers. You can build a new classifier on top of these that more accurately represents the new task.  


## Task

This repository contains a model that utilizes one of TensorFlow Kera's many built in pre-trained models, originally trained on the ImageNet dataset, to classify images in the CIFAR10 dataset using transfer learning methods.


### Pretrained Model

The model I used for this transfer learning task is the Inception-ResNet-V2. (why)

The Inception-ResNetV2 is a convolutional neural network that is based on the Inception network and its variants, but it also utilizes residual connections in place of the filter concatenation stage. The inception network architecture is used for it's computational efficency and high accuracy, while the residual connections help the network avoid the problem of the vanishing gradient.

(more info about inceptionresnetv2)

### Dataset
The CIFAR10 dataset is one of the most popular datasets used in machine learning research. It is typically used with a convolutional neural network for object recognition. It consists of 60,000 color images with a resolution of 32x32. There are 10 different classes in this dataset:

![image](https://github.com/mokasp/atlas-machine_learning/assets/125315163/a7627b87-c3f2-4b20-9037-73bba3f9bf96)

Each class contains 6,000 images, 5,000 training images and 1,000 test images

### Method

feature extraction, resizing, avg pooling, dropout, flatten, dense layers, batch norm, activation, softmax, learning rate decay, early stopping, batch size, shuffle


### Results



