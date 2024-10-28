import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].

 

	data_lowlight = (np.asarray(data_lowlight)/255.0) # Convert the image from Image object into Numpy array, then replace that image of Image object version with the Numpy array version one. Then, only normalize the pixel values (features) of that image to the range [0,1].


	data_lowlight = torch.from_numpy(data_lowlight).float()  # Convert the pixel values (features) of the image from Numpy array into tensor, with the data type of float32. So now the image becomes a 3D tensor.
	data_lowlight = data_lowlight.permute(2,0,1) # Use permute() to rearrange the dimensions of the image from [height(H),width(W),channels(C)] to [channels(C),height(H),width(W)]
	data_lowlight = data_lowlight.cuda().unsqueeze(0) # Add a new dimension of value 1 to the image at the 0th position of its tensor shape. So its tensor shape is changed from [channels,height,width] to [NumberOfSamplesInABatch(N),channels(C),height(H),width(W)], where NumberOfSamples(N),channels(C)=1 [representing it is a single image]. This unsqueeze must be performed to have the correct tensor shape that fits the input shape format of convolutional layer

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('D:/AI_Master_New/Low-Light_Image_Enhancemnet/Zero-DCE_code_LiChongYi_2021/snapshots/Epoch99.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'D:/AI_Master_New/Low-Light_Image_Enhancemnet/Zero-DCE_code_LiChongYi_2021/data/test_data/' # The absolute path that stores the test data

		resultPath = filePath.replace('test_data','result')
		# Create the folder called "DICM" in the folder "result" which is located at os.path.join(resultPath,"DICM/") [to store the enhanced DICM images] if the folder has not been created. If the folder has been created, ignore this line.
		os.makedirs(os.path.join(resultPath,"DICM/"), exist_ok=True)
		# Create the folder called "LIME" in the folder "result" which is located at os.path.join(resultPath,"LIME/") [to store the enhanced LIME images] if the folder has not been created. If the folder has been created, ignore this line.
		os.makedirs(os.path.join(resultPath,"LIME/"), exist_ok=True)

		file_list = os.listdir(filePath) # Returns a list containing the names of the entries in a directory specified by path. Here, the returned list is ['DICM', 'LIME'], the names of subfolder inside the folder "test_data". 

		for file_name in file_list: # For each subfolder inside the test_data folder (DCIM subfolder first, then only LIME subfolder)
			test_list = glob.glob(filePath+file_name+"/*") # Returns a list of files' absolute path (of any extension) that are inside the specified path (filePath+file_name).
			for image in test_list: # For each image, represented by its file's absolute path
				# image = image
				print(image) # Show the absolute path of that image
				lowlight(image) # Perform image processing and image enhancement using Zero-DCE on that image 


# Notes: 
# 1) Each BMP file has large file size because it contains uncompressed data, making it ideal for high-quality digital image.
# 2) NVIDIA GeForce MX150 failed to process(enhance) the BMP files because each of the bmp files has large file size, such that the VRAM cannot handle the data of each BMP file (especially the one with large file size)

