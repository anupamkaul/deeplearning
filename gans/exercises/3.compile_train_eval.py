
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10

print("\n\nLoading (or reading cached values of) CIFAR-10 dataset:\n")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0
x_test =  x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test =  to_categorical(y_test, NUM_CLASSES)

print("\ny_train :\n", y_train, "\n")

print ("sample pixel in x_train : ", x_train[54, 12, 13, 1], "\n")

"""

The 2nd example shows various ways of building the model
once data sets have been downloaded and pre-loaded into training 
& test tensors

In Keras there are 2 ways of defining the structure of my deep neural 
network (or 'building' them) : as a Sequential model or using the 
Functional API

Sequential model is useful for quickly defining a linear (only) stack of 
layers (i.e. one layer follows on directly from the previous w/o any branching. 

Many models though require that output from a layer is passed to multiple 
seperate layers beneath it, conversely that an input layer recieves inputs 
frm multiple layers above it (or juxtapose right and left terminologies)

So lets focus on Functional APIs which are more flexible, allow me to build from scratch (my model arch),
and serve me better inn the long run, as my neural networks become more architecturally complex. FunctionalAPI
gives me complete freedom over the design of my deep neural network.

"""

# Here is an example of making a model using Functional API
# Example 2.2 : Architecture using FunctionalAPI

from keras.layers import Input, Flatten, Dense
from keras.models import Model

input_layer = Input(shape=(32, 32, 3))

x = Flatten()(input_layer)

x = Dense(units=200, activation = 'relu')(x)
x = Dense(units=150, activation = 'relu')(x)

output_layer = Dense(units=10, activation = 'softmax')(x)

model = Model(input_layer, output_layer)

print("\nSummary of Model using Functional API:\n")
model.summary()

"""
Explanations for building the model:



"""

"""

3.The next step is: Compiling the model

We need a loss function and an optimizer

"""

print("\nCompiling the model (opt=Adam, loss=cat_cross-entropy, lr=0.005)\n)")
from keras.optimizers import Adam

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print(".. (done)")

"""

4. Training the model ! 

"""


print("\nTraining the model\n)")

model.fit(x_train,
          y_train,
          batch_size = 32,
          epochs = 10,
          shuffle = True
          ) 


"""
Explanations for training:

This will start training deep neural network to predict the category
of an image from a CIFAR-10 dataset


"""

"""

5. Evaluating the model

"""

print("\nEvaluating the model for ", model.metrics_names, "\n)")
model.evaluate(x_test, y_test)

"""
The output frm the above method should be a list of metrics we are monitoring:
categorical cross entropy and accuracy. 

Further we can view (visualize) some of the test set predictions using 
the predict method (and display using matplotlib)

"""

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                    'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)  # preds is an array of shape [10000, 10]
# (i.e. a vector of 10 class probabilities for each observation)

preds_single =  CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

# convert the array of probabilities back into a single prediction using numpy's argmax fn.
# axis = -1 tells the fn to collapse the array over the last dimension (the classes dimension),
# so that the shape of preds_single is then [10000, 1]

# Let's view some of the images along side their labels and prediction. We should not be expecting
# more than 50% accuracy (given the results reported):

print("\nDisplay some results with matplotlib..\n")

import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show) # take some random images from test...

# text the predictions as matplotlib doesn't render for now
print("random indices to show: ", indices, " \n")
for i, idx in enumerate(indices):
    print("\nFor index: ", i, "( image no. ", idx, ")\n")
    print("prediction  = ",  str(preds_single[idx]))
    print("actually is = ",  str(actual_single[idx]))

# show the images and plot the text in matplotlib:
fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax  = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')

    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10,
            ha='center', transform=ax.transAxes)

    ax.imshow(img)

plt.show()











