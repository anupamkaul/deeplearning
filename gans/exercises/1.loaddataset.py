"""
The 1st example shows how to download a dataset (CIFAR-10)
and how to extract training and test data for a neural network

"""

import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10

print("\n\nLoading (or reading cached values of) CIFAR-10 dataset:\n")

# read here: ~/miniconda3py38/lib/python3.8/site-packages/keras/datasets/cifar10.py for load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10

"""
(1) above: loads the CIFAR-10 dataset. x_train and x_test are numpy arrays of shape
[50000, 32, 32, 3] and [10000, 32, 32, 3] rsptvy. y_train and y_test are
numpy arrays with shape [50000, 1] and [10000, 1] rsp containing the integer
in the range 0 to 9 for the class of each image (10 classes, discriminator network)
"""

x_train = x_train.astype('float32') / 255.0
x_test =  x_test.astype('float32') / 255.0

"""
(2) above: by default image data of pixels is integers between 0 and 255 for each pixel
channel. Neural networks work best when each input is inside the range -1 to 1 (PDFs have sample inputs that add/integrate to 1)
So we need to divide by 255
"""

y_train = to_categorical(y_train, NUM_CLASSES)
y_test =  to_categorical(y_test, NUM_CLASSES)

"""
(3) above: we also need to change the integer labelling of the images to one-hot-encoded vectors.
If the class integer label of an image is i, then its one-hot-encoding is a vector of
length 10 (the number of classes) that has 0s in all but the ith element, which is 1.
The new shapes of y_train and y_test are therefore [50000, 10] and [10000, 10] rsp.
"""

"""
more:
the classes of images are : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

re: shape of x_train : [50000, 32, 32, 3]:
The first dim of the array references the index of the image in the dataset
second and third relate to the size of the image
last is the channel (i.e. red, green, blue) since these are RGB images.
there are no cols or rows in this dataset; instead, this is a "tensor" in 4 dims.

for example, the following entry refers to the green channel (1) value of the pixel in the (12, 13) position 
of image 54:
"""

print("\ny_train :\n", y_train, "\n")

print ("sample pixel in x_train : ", x_train[54, 12, 13, 1], "\n")

print("\ndone")





