'''

One of the reasons the model is modest at 50% accuracy is because it doesn't
exploit the spacial structure of the input images. We can use convolutional 
layers (CNN).

Text for this material is described from pages 46 to 51 of Generative Deep Learning

Convolution is performed by multiplying the filter (fig 2.10) pixelwise with the portion
of the image in question, and summing the result. The output is more positive when the
portion of the image closely matches the filter and negative when it is more of the inverse.
(This example is for a grayscale image, and filter is a simple 3 * 3 array with 1s and os
and multiplication is pixel * pixel and result added to a + or - value)

If we move the filter across the entire image, from left to right and top to bottom, recording
the convolutional output as we go, we obtain a new array that 'picks out' a particular feature
of the input, depending on the values of the filter (which are weights that get adjusted on every
back-prop of a batch of each epoch of training). 

A convolutional layer does exactly the above, but with Multiple filters instead of one. Fig 2.11
shows 2 filters applied to an image (detect vertical and horizontal edges). Similarly if we are
working with colored images then each filter would have 3 channels (hence multidimensional tensor)
rather than 1 (each having shape 3 * 3 * 3) to match the R, G, B channels of the image.

'''

'''
Prepare training data
'''

import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10

print("\n\nLoading (or reading cached values of) CIFAR-10 dataset:\n")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# to ever re-install, delete $HOME/.keras/datasets...

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0
x_test =  x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test =  to_categorical(y_test, NUM_CLASSES)

print("\ny_train :\n", y_train, "\n")
print ("sample pixel in x_train : ", x_train[54, 12, 13, 1], "\n")


'''
Build the model

In Keras the Conv2D layer applies convolutions to an input tensor with 2 spatial
dimensions (such as an image). 
'''

from keras.layers import Input, Flatten, Dense, Conv2D, LeakyReLU, BatchNormalization, Dropout, Activation
from keras.models import Model

# all of this code is in sub folders of /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras

#e.g. Conv2D             : /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/layers/convolutional/conv2d.py (class Conv2D)
#e.g. BatchNormalization : /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/layers/normalization/<file> (class) etc
#e.g. LeakyRLU           : /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/layers/activation/
#e.g. Dense              : /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/layers/core/
#e.g. Flatten            : /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/layers/reshaping/
#e.g. Adam               : /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/

# many of these files refer to the original arxiv papers they implement..
# specific to Keras, its concept of Layer is in /home/anupam/miniconda3py38/lib/python3.8/site-packages/keras/engine (base_layer(vX).py)

input_layer = Input((32,32,3))

x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.5)(x)

x = Dense(NUM_CLASSES)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)
model.summary()


'''
Compile the model as before, using Adam for Opt and Cross Entropy for loss
'''

print("\nCompiling the model (opt=Adam, loss=cat_cross-entropy, lr=0.005)\n")
from keras.optimizers import Adam

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print(".. (done)")

'''
Train the model, as before
'''

print("\nTraining the model\n")

model.fit(x_train,
          y_train,
          batch_size = 32,
          epochs = 10,
          shuffle = True,
          validation_data = (x_test, y_test)
          ) 

'''
Evaluate the model on test data
'''

print("\nEvaluating the model for ", model.metrics_names, "\n)")
print(model.evaluate(x_test, y_test, batch_size=1000))

