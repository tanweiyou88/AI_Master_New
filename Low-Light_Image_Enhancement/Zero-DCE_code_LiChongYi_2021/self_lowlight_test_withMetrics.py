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

import numpy as np
import pandas as pd

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from tqdm import tqdm
import torch.utils.data


def image_enhancement():
	start = time.time() # Get the starting time of image enhancement process on a batch of samples

	_,enhanced_image,_ = DCE_net(img_lowlight) # Perform image enhancement using Zero-DCE on the current batch of images/samples

	iteration_duration = (time.time() - start) # Get the duration of image enhancement process on the current batch of samples

	print("\nDuration of image enhancement process on the current batch of samples:", iteration_duration)

	return enhanced_image

def save_image():
	
	resultPath = testPath.replace('test_data','result') # Change the path
	resultPath_subfolder = "self_CompileMetrics" # Define the folder that stores the grids of input-output image pairs of this script

	# Create the path that stores the grids of input-output image pairs of this script
	os.makedirs(os.path.join(resultPath, resultPath_subfolder), exist_ok=True)
	
	# Create a grid of input-output image pairs involved in the current iteration 
	grid_image = torchvision.utils.make_grid(torch.cat((img_lowlight, enhanced_image), 0), nrow=img_lowlight.shape[0])
	
	# Save the created grid of input-output image pairs involved in the current iteration 
	torchvision.utils.save_image(
		grid_image,
		os.path.join(resultPath, resultPath_subfolder, '{}.jpg'.format(iteration + 1))
	)
	
def metrics_calculation(LLIE_metrics_data): # We define LLIE_metrics_data as the input argument, so that the results of LLIE_metrics_data calculated in this self defined function can be directly updated to the LLIE_metrics_data located in the main() of this script
	sample_size = img_lowlight.shape[0] # Get the total number of img_lowlight samples in current iteration, can be equivalent to batch size
	LLIE_metrics_data['accumulate_number_of_samples_processed'] += sample_size # Update the img_lowlight samples processed in the current iteration. So that at the end, LLIE_valing_results['batch_sizes'] stores the total number of img_lowlight samples that has been processed by the model.

	batch_average_psnr = psnr(enhanced_image, img_lowlight).item() # torchmetrics.PeakSignalNoiseRatio first calculates the PSNR of each image pair in a batch of images, then return the average PSNR of all image pairs in that batch. The mathematical concept behinds is: (average PSNR)=([Summation of PSNR of all image pairs in a batch of images]/[Total number of image pairs in that batch of images, which is same as the batch size]) of that batch of images.
	batch_average_ssim = ssim(enhanced_image, img_lowlight).item()
	LLIE_metrics_data['accumulate_psnr'] += batch_average_psnr * sample_size # Get the total PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (batch_psnr * cfg.batch_size)=(Summation of PSNR of all image pairs in a batch of images).
	LLIE_metrics_data['accumulate_ssim'] += batch_average_ssim * sample_size

	LLIE_metrics_data['average_psnr'] = LLIE_metrics_data['accumulate_psnr'] / LLIE_metrics_data['accumulate_number_of_samples_processed'] # Get the average PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (Summation of PSNR of all image pairs the model has gone through)/[Total batches of images the model has gone through, which is same as the total number of image pairs the model has gone through]).
	LLIE_metrics_data['average_ssim'] = LLIE_metrics_data['accumulate_ssim'] / LLIE_metrics_data['accumulate_number_of_samples_processed']
	
	batch_average_lpips = lpips(enhanced_image, img_lowlight).item()
	LLIE_metrics_data['accumulate_lpips'] += batch_average_lpips * sample_size
	LLIE_metrics_data['average_lpips'] = LLIE_metrics_data['accumulate_lpips'] / LLIE_metrics_data['accumulate_number_of_samples_processed']
	print('[LLIE] Accumulated processed sample numbers:%d, Average PSNR: %.4f dB, Average SSIM: %.4f, Average LPIPS: %.4f' % (
                        LLIE_metrics_data['accumulate_number_of_samples_processed'], LLIE_metrics_data['average_psnr'], LLIE_metrics_data['average_ssim'], LLIE_metrics_data['average_lpips']))
	
	


if __name__ == '__main__':

	parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
	# Input Parameters
	parser.add_argument('--lowlight_images_test_path', type=str, default="D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/test_data/") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	# parser.add_argument('--lr', type=float, default=0.0001) # Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	# parser.add_argument('--weight_decay', type=float, default=0.0001)
	# parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	# parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--test_batch_size', type=int, default=8)
	# parser.add_argument('--val_batch_size', type=int, default=4) # Original batch size
	parser.add_argument('--num_workers', type=int, default=4)
	# parser.add_argument('--display_iter', type=int, default=10)
	# parser.add_argument('--snapshot_iter', type=int, default=10)
	# parser.add_argument('--snapshots_folder', type=str, default="D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/snapshots/self_train_snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= True)
	parser.add_argument('--pretrain_dir', type=str, default= "D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/snapshots/Epoch99.pth")
	# The parse_args() object (config=self) takes the data (values) you provide to your positional/optional arguments on command line interface or within the () of parse_args(), then converts them into the required data type as mentioned in add_argument() respectively. 
	# So you can access the data of a positional/optional argument by using the syntax args.argument_name (EG: config.lowlight_images_path).
	config = parser.parse_args() 

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	
	ssim = SSIM(data_range=1.).to(device) # Performance metric: SSIM
	psnr = PSNR(data_range=1.).to(device) # Performance metric: PSNR
	lpips = LPIPS(net_type='alex').to(device) # Performance metric: LPIPS (perceptual loss provided by a pretrained learning-based model), using alexnet as the backbone (LPIPS with alexnet performs the best according to its official Github)
	LLIE_metrics_data = {'mse': 0, 'accumulate_ssim': 0, 'accumulate_psnr': 0, 'average_psnr': 0, 'average_ssim': 0, 'accumulate_number_of_samples_processed': 0, 'accumulate_lpips': 0, 'average_lpips': 0,} # Initialize the metrics, so they can update from time to time later

# test_images	
	with torch.no_grad():
		
		DCE_net = model.enhance_net_nopool().to(device) # Move the model to cuda so that the samples in each batch can be processed simultaneously
		DCE_net.eval() # Set the model to evaluation mode

		if config.load_pretrain == True: # When NOT set to train mode 
			DCE_net.load_state_dict(torch.load(config.pretrain_dir)) # load the parameters (weights & biases) of the model obtained/learned at the snapshot (EG: at a particular epoch)

		testPath =  config.lowlight_images_test_path # The absolute path that stores the test data
		test_list = os.listdir(testPath) # Returns a list containing the names of the entries in a directory specified by path. Here, the returned list is ['DICM', 'LIME'], the names of subfolder inside the folder "test_data". 
		
		for testfile_name in test_list: # For each subfolder inside the test_data folder (DCIM subfolder first, then only LIME subfolder)
			if testfile_name == "DICM": # Only takes the DICM test data available in testPath
				test_path = testPath + testfile_name +"/"
				print("test_path:", test_path)
				test_dataset = dataloader.lowlight_loader(test_path) # Create custom test dataset: Take the images available in test_path, then preprocess them, and convert them into a tensor
		
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True) # Split the samples in the test dataset into batches/groups/collections of test_batch_size samples. By default, test_batch_size=8, that's means each batch has 8 samples.
		test_bar = tqdm(test_loader) # tqdm automatically detects the length of the iterable object (train_loader) [The length of the iterable object = The total number of batches of samples] and generates a progress bar that updates dynamically as each item=(batch of samples) is processed.
		
		for iteration, img_lowlight in enumerate(test_bar):
			img_lowlight = img_lowlight.to(device) # Since this batch of train_batch_size samples are moved to cuda for processinng, they will be processed simultaneously.
			
			enhanced_image = image_enhancement() # Perform image enhancement on the current batch of samples using the model
			
			metrics_calculation(LLIE_metrics_data) # Perform metric calculations
			save_image() # Save the grid of input-output image pairs involved in the current iteration

		print('\n----------------------------------Final results of LLIE----------------------------------')
		print('Total processed sample numbers:%d, Average PSNR: %.4f dB, Average SSIM: %.4f, Average LPIPS: %.4f' % (
                        LLIE_metrics_data['accumulate_number_of_samples_processed'], LLIE_metrics_data['average_psnr'], LLIE_metrics_data['average_ssim'], LLIE_metrics_data['average_lpips']))
	
				

# Notes: 
# Study when is really need the input arguments for a self defined function
# Add functions to calculate params, FLOPs, and average runtime (Duration of image enhancement process)

