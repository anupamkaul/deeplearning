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

print("\nSummary of Model using Sequential construct:\n")
model.summary()

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

print("\nCompiling the model (opt=Adam, loss=cat_cross-entropy, lr=0.005))\n")
from keras.optimizers import Adam

from sys import platform

if platform == "linux" or platform == "linux2":
	print("run adam on linux\n")
	opt = Adam(lr=0.005)
elif platform == "darwin": #OS-X
	print("run adam on mac\n")
	opt = Adam(learning_rate=0.005)
else:
	print("hmm ... new OS, not running Adam yet\n")

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print(".. (done)")








