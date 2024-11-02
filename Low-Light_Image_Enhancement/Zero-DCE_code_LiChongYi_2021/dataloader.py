import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

# **Important information:**
# 1. dataloader.py contains functions (dataset code) to create a custom dataset (by loading the files/samples from the specified folder) + preprocess the samples in the created custom dataset.
# 2. It is a convention to have the dataset code to be decoupled from the model training code for better readability and modularity.

random.seed(1143) # The random number generator (random) is seeded/initialized (random.seed(x)) with value/fixed state of 1143. So all random operations here return the reproducible results.

def populate_train_list(lowlight_images_path):




	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg") # Returns a list of files' absolute path (of .jpg extension) that are inside the specified path (lowlight_images_path)

	train_list = image_list_lowlight # train_list stores a list of images' absolute path

	random.shuffle(train_list) # randomly shuffle the elements of a list (image_list_lowlight), altering the sequence in place. Since the random number generator (random) is seeded (random.seed(x)), the random shuffled list is reproducible.

	return train_list


# **Self add**
# Creating a Custom Dataset for your files: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# -> A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.

# The lowlight_loader class is used to create Custom Dataset
class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path): # The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the samples (Here, the samples are the images)

		self.train_list = populate_train_list(lowlight_images_path) # Return a list of images' absolute path inside a folder, after randomly shuffling them
		self.size = 256 # The width and height of each image, in pixel

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index): # The __getitem__ function loads and returns a sample from the dataset at the given index.

		data_lowlight_path = self.data_list[index] # Return a single image's absolute path that is located at index on the data_list
		
		data_lowlight = Image.open(data_lowlight_path) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
		
		#data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS) # Resize the opened image to (self.size,self.size) with Image.ANTIALIAS filter for anti-aliasing, using PIL. Then, replace the opened image with its resized image at the given image's absolute path.
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.LANCZOS) # Resize the opened image to (self.size,self.size) with Image.ANTIALIAS filter for anti-aliasing, using PIL. Then, replace the opened image with its resized image at the given image's absolute path. **ANTIALIAS was removed in Pillow 10.0.0 (after being deprecated through many previous versions). Now you need to use PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.(This is the exact same algorithm that ANTIALIAS referred to, you just can no longer access it through the name ANTIALIAS.)**

		data_lowlight = (np.asarray(data_lowlight)/255.0) # Convert the resized image from Image object into Numpy array, then replace the resize image of Image object version with the Numpy array version one. Then, only normalize the pixel values (features) of the resized image to the range [0,1].
		data_lowlight = torch.from_numpy(data_lowlight).float() # Convert the pixel values (features) of the resized image from Numpy array into tensor, with the data type of float32.

		return data_lowlight.permute(2,0,1) # Use permute() to rearrange the dimensions of the image from [height(H),width(W),channels(C)] to [channels(C),height(H),width(W)]

	def __len__(self): # The __len__ function returns the number of samples in our dataset.
		return len(self.data_list)

