
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

The 2nd example shows various ways of building the model
once data sets have been downloaded and pre-loaded into training 
& test tensors

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

Using the Functional API, here we are using 3 types of layers : Input, Flatten and Dense.
Input is an entry point into the network. We tell the network the shape of each data element
to expect as a tuple. Notice that we don't specify the batch size (not necessary) as we can
pass any number of images into the input layer simultaneously. 

Next we Flatten the input into a vector using Flatten layer. This results in a vector of 
length 32 * 32 * 3 = 3072. Reason we do this here is because the Dense layer needs a flat input
(instead of a multi-dim array). (However other models'Dense layers  may require multi-dim array inputs)

Dense layer is fundamental to a neural network. It contains a number of units that are densly connected
to the prev layer, i.e. every unit in this layer is connected to every input in the previous layer, thru
a single connection that carries a weight (which can be positive or negative). The output from a given
unit is the weighted sum of the input it recieves from the previous layer to the following layer, which
is then passed to a nonlinear Activation Function before being sent to the following layer. The activation
function is critical to ensure the neural network is able to learn complex functions and doesn't just output
a linear combination of its input. 

There are many kinds of activation function but 3 most important are ReLU, Sigmoid, and Softmax.

ReLU (Recitified Linear Unit) activiation function is defined to be zero if i/p is negative and otherwise
equal to the input. LeakyRLU is a variation where whereas ReLU returns zero for i/p values less than zero,
LeakyRLU fn returns a small negative number proportional to the i/p/ ReLU units can sometimes die if they
always output zero, bcos of a large bias toward negative values preactivation. In this case gradient is zero
and therefore no error is propagated back through this unit. LeakyReLU activations fix this issue by always 
ensuring that gradient is non-zero. ReLU based activation functions are now considered the most reliable 
activations to use between the layers of a deep network to encourage stable training.

Sigmoid is useful for scaling output of a layer to be between 0 and 1. (e.g. for binary classification
with 1 o/p unit or multilabel classification problems) where each output can belong to MORE than 1 class. 
(y = 1 / (1 + e(-x)))

Softmax is useful when you want the total SUM of the output layer to be equal to 1, for example, 
for multiclass classification problems where each observation ONLY BELONGS TO EXACTLY 1 CLASS. 
(Yi = e(Xi) / (Sigma(j=1, J)(e(Xj))), where J is the total number of units in a layer.

In our neural network we use a softmax activation in the final layer to ensure that the output is a set 
of 10 probabilities that sum to 1, which can be interpreted that the image belongs to each class.

In our example model, we pass the input through 2 dense hidden layers, the first with 200 units, the second 
with 150 units, both with ReLU activation functions. Final step is the define the model itself using Model
class. We can also define Model with multiple input and output layers if needed (to be seen later). In our
model the shape of Input layer matches the shape of x_train, and shape of Dense layer matches shape of 
y_train. Model.summary9) displays shape of each layer in the network.

"""

print("\nCompiling the model (opt=Adam, loss=cat_cross-entropy, lr=0.005)\n")
from keras.optimizers import Adam

opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print(".. (done)")

"""

3.The next step is: Compiling the model

We need a loss function and an optimizer

Optimizer notes:
---------------

Using ADAM (Adaptive Moment Estimation) as the optimizer (https://arxiv.org/pdf/1412.6980.pdf)
(An optimizer technique for gradient descent. Intuition: https://www.geeksforgeeks.org/intuition-of-adam-optimizer/)

Adam optimizer involves a combination of two gradient descent methodologies: 

(A) Momentum: 

This algorithm is used to accelerate the gradient descent algorithm by taking into consideration 
the ‘exponentially weighted average’ of the gradients. Using averages makes the algorithm converge towards 
the minima in a faster pace. 

w_{t+1}=w_{t}-\alpha m_{t}
where,

m_{t}=\beta m_{t-1}+(1-\beta)\left[\frac{\delta L}{\delta w_{t}}\right]
mt = aggregate of gradients at time t [current] (initially, mt = 0)
mt-1 = aggregate of gradients at time t-1 [previous]
Wt = weights at time t
Wt+1 = weights at time t+1
αt = learning rate at time t 
∂L = derivative of Loss Function
∂Wt = derivative of weights at time t
β = Moving average parameter (const, 0.9)


(B): Root Mean Square Propagation (RMSP): 

Root mean square prop or RMSprop is an adaptive learning algorithm that tries to improve AdaGrad. 
Instead of taking the cumulative sum of squared gradients like in AdaGrad, it takes the ‘exponential moving average’.

w_{t+1}=w_{t}-\frac{\alpha_{t}}{\left(v_{t}+\varepsilon\right)^{1 / 2}} *\left[\frac{\delta L}{\delta w_{t}}\right]

where, 

v_{t}=\beta v_{t-1}+(1-\beta) *\left[\frac{\delta L}{\delta w_{t}}\right]^{2}
Wt = weights at time t
Wt+1 = weights at time t+1
αt = learning rate at time t 
∂L = derivative of Loss Function
∂Wt = derivative of weights at time t
Vt = sum of square of past gradients. [i.e sum(∂L/∂Wt-1)] (initially, Vt = 0)
β = Moving average parameter (const, 0.9)
ϵ = A small positive constant (10-8)

"""

"""

4. Training the model ! 

"""


print("\nTraining the model\n")

model.fit(x_train,
          y_train,
          batch_size = 32,
          epochs = 10,
          shuffle = True
          ) 


"""
Explanations for training:

This will start training deep neural network to predict the category
of an image from a CIFAR-10 dataset. We need the training data (x_train),
(this is the raw image data), y_train is the one-hot encoded class labels associated (supervised learning),

Batch_size determines how many observations will be passed to the network at each training step.
Epochs determine how many times the network will be shown full data. Shuffle=true means that batches will be
drawn randomly without replacement from the training data at each training step.

Training proceeds as follows:

First, weights of the network are initialized to random values. Then the network performs a series of training steps.

At each training step, one batch of images is passed through the network and the errors are back propagated to update
the weights. Batch_size determines how many images are shown in each training step batch. Larger the batch size, more
stable the gradient but slower the training step. Batch size = entire data set is too computationally intensive today
so batch size between 32 and 256 is used. Recommended practice now is to increase the batch size as training progresses.

This continues until all observations in the dataset have been seen once & weights have been re-evaluated. This constitutes
1 epoch. Data is then passed again through the n/w in the next epoch. Process repeats until the number of epochs have elapsed.
Output is metrics (in this case accuracy of prediction - tested via eval new data, and loss factor here being categorial cross entropy).



"""

"""

5. Evaluating the model

"""

print("\nEvaluating the model for ", model.metrics_names, "\n)")
print(model.evaluate(x_test, y_test))

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











