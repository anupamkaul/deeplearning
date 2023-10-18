#ref https://matplotlib.org/stable/tutorials/images.html

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

# import image data into numpy arrays

'''
Matplotlib relies on the Pillow library to load image data.

stinkbug.png: 
it's a 24-bit RGB PNG image (8 bits for each of R, G, B). 
Depending on where you get your data, the other kinds of image 
that you'll most likely encounter are RGBA images, which allow 
for transparency, or single-channel grayscale (luminosity) images. 

We use Pillow to open an image (with PIL.Image.open), 
and immediately convert the PIL.Image.Image object into an 8-bit (dtype=uint8) numpy array.

'''

img = np.asarray(Image.open('./stinkbug.png'))
print(repr(img)) 

'''
array([[[104, 104, 104],
        [104, 104, 104],
        [104, 104, 104],
        ...,
        [109, 109, 109],
        [109, 109, 109],
        [109, 109, 109]],

       [[105, 105, 105],
        [105, 105, 105],
        [105, 105, 105],
        ...,
        [109, 109, 109],
        [109, 109, 109],
        [109, 109, 109]],

       [[107, 107, 107],
        [106, 106, 106],
        [106, 106, 106],
        ...,
        [110, 110, 110],
        [110, 110, 110],
        [110, 110, 110]],

       ...,

       [[112, 112, 112],
        [111, 111, 111],
        [110, 110, 110],
        ...,
        [116, 116, 116],
        [115, 115, 115],
        [115, 115, 115]],

       [[113, 113, 113],
        [113, 113, 113],
        [112, 112, 112],
        ...,
        [115, 115, 115],
        [114, 114, 114],
        [114, 114, 114]],

       [[113, 113, 113],
        [115, 115, 115],
        [115, 115, 115],
        ...,
        [114, 114, 114],
        [114, 114, 114],
        [113, 113, 113]]], dtype=uint8)

Each inner list represents a pixel. 
Here, with an RGB image, there are 3 values. 
Since it's a black and white image, R, G, and B are all similar. 
An RGBA (where A is alpha, or transparency) has 4 values per inner list, 
and a simple luminance image just has one value (and is thus only a 2-D array, not a 3-D array). 
For RGB and RGBA images, Matplotlib supports float32 and uint8 data types. 
For grayscale, Matplotlib supports only float32. 
If your array data does not meet one of these descriptions, you need to rescale it.

(good to know as this is an output of the latent space as well, later for Generative AI)

So, you have your data in a numpy array (either by importing it, or by generating it). 
Let's render it. 
In Matplotlib, this is performed using the imshow() function. 
Here we'll grab the plot object. This object gives you an easy way to manipulate the plot from the prompt.

'''

imgplot = plt.imshow(img)
plt.show() #needed to add this for popup to show where imshow's o/p is rendered

text = input("enter something") # wait for user input because imshow didn't seem to render

# can run some additional experiments on the image now per the tutorial


