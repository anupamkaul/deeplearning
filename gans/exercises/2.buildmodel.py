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

So lets focus on Functional APIs which are more flexible, allow you to build from scratch your model arch,
and serve me better inn the long run, as my neural networks become more architecturally complex. FunctionalAPI
gives me complete freedom over the design of my deep neural network.

"""

# Here are 2 examples that show usage of Sequential model and Functional API

# Example 2.1 : Architecture using a Sequential Model

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential([

    Dense(200, activation = "relu", input_shape=(32, 32, 3)),
    Flatten(),
    Dense(150, activation = "relu"),
    Dense(10,  activation = "softmax"),

])

model.summary()





