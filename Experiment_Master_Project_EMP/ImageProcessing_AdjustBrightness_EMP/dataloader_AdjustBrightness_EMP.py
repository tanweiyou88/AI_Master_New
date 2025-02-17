import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image
import glob
import random
import cv2

# **Important information:**
# 1. dataloader.py contains functions (dataset code) to create a custom dataset (by loading the files/samples from the specified folder) + preprocess the samples in the created custom dataset.
# 2. It is a convention to have the dataset code to be decoupled from the model training code for better readability and modularity.



# **Self add**
# Creating a Custom Dataset for your files: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# -> A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.

class image_loader(data.Dataset):

	# def __init__(self, images_folder_path, img_size=256, low_light_brightness_factor=0.3, high_light_brightness_factor=1.6): # The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the samples (Here, the samples are the images)
	def __init__(self, images_folder_path, img_size=256): # The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the samples (Here, the samples are the images)

		self.image_list = sorted(glob.glob(images_folder_path + "/*"), key=len) # Get a list of absolute file paths that available in the images_folder_path, in ascending sequence according to the length of each absolute file path
		# self.transforms = transforms
		self.size = img_size # The width and height of each image, in pixel
		# self.low_light_brightness_factor = low_light_brightness_factor
		# self.high_light_brightness_factor = high_light_brightness_factor

		print("Total images from source dataset folder path:", len(self.image_list))


	def __getitem__(self, index): # The __getitem__ function loads and returns a sample from the dataset at the given index.

		image_path = self.image_list[index] # Return a single image's absolute path that is located at index on the data_list
		
		image = Image.open(image_path) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
		if image.mode != "RGB": # If the image is not a RGB image
			image = image.convert("RGB") # Convert the image into RGB image
		image = image.resize((self.size,self.size), Image.LANCZOS) # Resize the opened image to (self.size,self.size) with Image.ANTIALIAS filter for anti-aliasing, using PIL. Then, replace the opened image with its resized image at the given image's absolute path. **ANTIALIAS was removed in Pillow 10.0.0 (after being deprecated through many previous versions). Now you need to use PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.(This is the exact same algorithm that ANTIALIAS referred to, you just can no longer access it through the name ANTIALIAS.)**
		image = (np.asarray(image)/255.0) # Convert the resized image from Image object into Numpy array, then replace the resize image of Image object version with the Numpy array version one. Then, only normalize the pixel values (features) of the resized image to the range [0,1].
		image = torch.from_numpy(image).float() # Convert the pixel values (features) of the resized image from Numpy array into tensor, with the data type of float32.
		
		return image_path, image.permute(2,0,1) # Return the file (image) absolute path and its image pixel information (in the format of [channels(C),height(H),width(W)])


	# def __getitem__(self, index): # The __getitem__ function loads and returns a sample from the dataset at the given index.

	# 	image_path = self.image_list[index] # Return a single image's absolute path that is located at index on the data_list
		
	# 	image = Image.open(image_path) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
	# 	image = image.resize((self.size,self.size), Image.LANCZOS) # Resize the opened image to (self.size,self.size) with Image.ANTIALIAS filter for anti-aliasing, using PIL. Then, replace the opened image with its resized image at the given image's absolute path. **ANTIALIAS was removed in Pillow 10.0.0 (after being deprecated through many previous versions). Now you need to use PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.(This is the exact same algorithm that ANTIALIAS referred to, you just can no longer access it through the name ANTIALIAS.)**
	# 	image = (np.asarray(image)/255.0) # Convert the resized image from Image object into Numpy array, then replace the resize image of Image object version with the Numpy array version one. Then, only normalize the pixel values (features) of the resized image to the range [0,1].
	# 	image = torch.from_numpy(image).float() # Convert the pixel values (features) of the resized image from Numpy array into tensor, with the data type of float32.
		
	# 	sample = {'image_path':image_path, 'image':image.permute(2,0,1)} # Each time return the image absolute path and its image pixel information, in the form of dictionary
		
	# 	return sample

	def __len__(self): # The __len__ function returns the number of samples in our dataset.
		return len(self.image_list)

