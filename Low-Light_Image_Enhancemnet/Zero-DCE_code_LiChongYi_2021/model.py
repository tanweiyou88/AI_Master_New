import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

# **Extra information: The relationship between the format of defining a convolutional layer and the format of input passing to the defined covolutional layer that is built on PyTorch**
# 1. When we define a convolutional layer on PyTorch, from the input data/samples [which are stored in a tensor] shape perspective, 
# we just define the number of channels of each input sample (C_in) and the number of channels of each output sample (C_out)
# (omitting the batch size [number of samples in a batch,N], height of each input sample (H_in), and width of each input sample (W_in)).
# 2. The input to that convolutional layer must be a tensor that has a shape of [NumberOfSamplesInABatch(N),channels(C_in),height(H_in),width(W_in)].
# 3. So after we create a custom dataset of samples/images, preprocessing the dataset, and before we feed the dataset to the model built on PyTorch for operations (EG: training) [EG: enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)], 
# we need to ensure the dataset is stored as a tensor that has a shape of [NumberOfSamplesInABatch(N),channels(C_in),height(H_in),width(W_in)], 
# and must follow the dimension order [because each dimension represents a unique meaning/information].

class enhance_net_nopool(nn.Module): # Define a class for the Zero-DCE model

	# Define the layers of the Zero-DCE model
	def __init__(self):
		super(enhance_net_nopool, self).__init__() # Execute the "def __init__" of the parent class (nn.Module)

		self.relu = nn.ReLU(inplace=True) # Define the ReLU layer

		number_f = 32 # The output channels (number of output feature maps = curve parameter maps) of a convolutional layer
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) # Define a convolutional layer that takes each sample/image of 3 channels(RGB image) and outputs number_f channels=numbers of output feature maps. Each kernel=filter of this convolutional layer has size of 3x3 pixels, stride of 1 and padding of 1. These kernel size, stride and padding settings are chosen to ensure each output feature map of this convolutional layer has the same dimension as its input channel (imitate the effect of padding=same in Tensorflow). Adds a learnable bias when perform each convolutional operation.
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) # Define a convolutional layer that takes each sample consisting of number_f channels=numbers of feature maps and outputs number_f channels=numbers of output feature maps. Each kernel=filter of this convolutional layer has size of 3x3 pixels, stride of 1 and padding of 1. These kernel size, stride and padding settings are chosen to ensure each output feature map of this convolutional layer has the same dimension as its input channel (imitate the effect of padding=same in Tensorflow). Adds a learnable bias when perform each convolutional operation.
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) # Define a convolutional layer that takes each sample consisting of number_f*2 channels=numbers of feature maps and outputs number_f channels=numbers of output feature maps. Each kernel=filter of this convolutional layer has size of 3x3 pixels, stride of 1 and padding of 1. These kernel size, stride and padding settings are chosen to ensure each output feature map of this convolutional layer has the same dimension as its input channel (imitate the effect of padding=same in Tensorflow). Adds a learnable bias when perform each convolutional operation.
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) # Define a convolutional layer that takes each sample consisting of number_f*2 channels=numbers of feature maps and outputs 24 channels=numbers of output feature maps. Each kernel=filter of this convolutional layer has size of 3x3 pixels, stride of 1 and padding of 1. These kernel size, stride and padding settings are chosen to ensure each output feature map of this convolutional layer has the same dimension as its input channel (imitate the effect of padding=same in Tensorflow). Adds a learnable bias when perform each convolutional operation.

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


	# Define the connection between each layer of the Zero-DCE model	
	def forward(self, x): # x will be a 4D tensor [batch_size(N), channels_of_each_image(C_in), height_of_each_image(H_in), width_of_each_image(W_in)] storing a batch of images

		x1 = self.relu(self.e_conv1(x)) # Pass a batch of images stored in x to the first convolutional layer. The output of the first convolutional layer will get passed to the ReLU layer. The output of the ReLU will then get stored in x1.
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1))) # Concatenate the feature maps stored in x3 and x4 along their 2nd dimension (channels dimension) sample-wise (The feature maps stored in x3 and x4 that corresponds to sample 1 is concatenated, and vice versa), so that they are now stacked together vertically as a single 4D tensor [performs concatenation-based skip connection]. The 4D tensor that stores the concatenated feature maps is passed to the fifth convolutional layer. The output of the fifth convolutional layer will get passed to the ReLU layer. The output of the ReLU will then get stored in x5.
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1))) # x_r stores 24 pieces of curve parameter maps = output 24-channels feature maps at the convolutional layer 7 (last layer)
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1) # x_r is splitted into 8 r (along their 2nd dimension [channels dimension]), such that each r consists of 3 curve parameter maps (Each curve parameter map for a color channel of the given image, since a color image has 3 color channels [RGB]).

		# Perform 8 iterations of image enhancement on a batch of images simultaneuosly. Each iteration use a unique r (unique set of 3 curve parameter maps, each curve parameter map for a channel of each image)
		x = x + r1*(torch.pow(x,2)-x) # Enhancement iteration 1: Iteration 1 of enhancing the given image using r1 that stores 3 curve parameter maps
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)	# enhance_image_1 stores a batch of the enhanced images obtained after the 4th iteration of enhancement.	
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x) # Enhancement iteration 8: Iteration 8 of enhancing the given image using r8 that stores 3 curve parameter maps. enhance_image stores a batch of the final version of enhanced images generated by the Zero-DCE model.	
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1) # r is a 4D tensor that stores all 24 curve parameter maps which concatenated along their channel dimension (1 curve parameter map at 1 channel) = x_r
		return enhance_image_1,enhance_image,r



