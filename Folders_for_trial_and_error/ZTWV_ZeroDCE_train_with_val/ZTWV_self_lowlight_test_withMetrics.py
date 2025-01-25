import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
# import sys
import argparse
import time
import ZTWV_dataloader
import ZTWV_model
import numpy as np
from torchvision import transforms
from PIL import Image
# import glob
import time

import numpy as np
# import pandas as pd

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE
from tqdm import tqdm
import torch.utils.data

from calflops import calculate_flops # for calflops to calculate MACs, FLOPs, and trainable parameters
import csv


def image_enhancement(IQA_metrics_data, ComputationalComplexity_metrics_data): # We define IQA_metrics_data and ComputationalComplexity_metrics_data as the input arguments, so that the results related to IQA_metrics_data and ComputationalComplexity_metrics_data calculated in this self-defined function can be directly updated to the corresponding metric dictionaries located in the main() of this script.
	start = time.time() # Get the starting time of image enhancement process on a batch of samples, in the unit of second.

	_,enhanced_image,_ = DCE_net(img_lowlight) # Perform image enhancement using Zero-DCE on the current batch of images/samples. enhanced_image stores the enhanced images.

	batch_duration = (time.time() - start) # Get the duration of image enhancement process on the current batch of samples
	IQA_metrics_data['batch_duration'] = batch_duration # Update the current batch duration to the IQA_metrics_data dictionary
	ComputationalComplexity_metrics_data['accumulate_batch_duration'] += IQA_metrics_data['batch_duration'] # Update the accumulate batch duration to the ComputationalComplexity_metrics_data dictionary
	print("\nDuration of image enhancement process on the current batch of input samples [second (s)]:", IQA_metrics_data['batch_duration'])

	return enhanced_image

def save_image():
	# Save the enhanced images involved in the current iteration 
	for count, image in enumerate(enhanced_image):
		sample_size = enhanced_image.shape[0] # Get the total number of enhanced_image samples in current iteration, can be equivalent to batch size
		# Save each individual enhanced image, according to the sequence being processed by the model
		torchvision.utils.save_image(
			image,
			os.path.join(resultPath_EnhancedResults, 'enhanced_image_{}.jpg'.format( IQA_metrics_data['accumulate_number_of_input_samples_processed'] - sample_size + count + 1) )
		)

	# Create a grid of input-output image pairs involved in the current iteration 
	grid_image = torchvision.utils.make_grid(torch.cat((img_lowlight, enhanced_image), 0), nrow=img_lowlight.shape[0])
	
	# Save the created grid of input-output image pairs involved in the current iteration 
	torchvision.utils.save_image(
		grid_image,
		os.path.join(resultPath_ImagePairsResults, 'InputEnhanced_ImagePair_{}.jpg'.format(iteration + 1))
	)
	
def IQA_metrics_calculation(IQA_metrics_data): # We define IQA_metrics_data as the input argument, so that the results related to IQA_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	sample_size = img_lowlight.shape[0] # Get the total number of input samples (img_lowlight.shape[0]) that have been processed by the model in the current iteration, can be equivalent to batch size.
	IQA_metrics_data['accumulate_number_of_input_samples_processed'] += sample_size # Update the processed input samples in the current iteration. So that at the end, IQA_metrics_data['accumulate_number_of_input_samples_processed'] stores the total number of input samples that have been processed by the model.

	# PSNR part
	batch_average_psnr = psnr(enhanced_image, img_lowlight).item() # torchmetrics.PeakSignalNoiseRatio first calculates the PSNR of each image pair in a batch of images, then return the average PSNR of all image pairs in that batch. The mathematical concept behinds is: (average PSNR)=([Summation of PSNR of all image pairs in a batch of images]/[Total number of image pairs in that batch of images, which is same as the batch size]) of that batch of images.
	IQA_metrics_data['accumulate_psnr'] += batch_average_psnr * sample_size # Get the total PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (batch_psnr * cfg.batch_size)=(Summation of PSNR of all image pairs in a batch of images).
	IQA_metrics_data['average_psnr'] = IQA_metrics_data['accumulate_psnr'] / IQA_metrics_data['accumulate_number_of_input_samples_processed'] # Get the average PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (Summation of PSNR of all image pairs the model has gone through)/[Total batches of images the model has gone through, which is same as the total number of image pairs the model has gone through]).

	# SSIM part
	batch_average_ssim = ssim(enhanced_image, img_lowlight).item()
	IQA_metrics_data['accumulate_ssim'] += batch_average_ssim * sample_size
	IQA_metrics_data['average_ssim'] = IQA_metrics_data['accumulate_ssim'] / IQA_metrics_data['accumulate_number_of_input_samples_processed']

	# MAE part
	batch_average_mae = mae(enhanced_image, img_lowlight).item()
	IQA_metrics_data['accumulate_mae'] += batch_average_mae * sample_size
	IQA_metrics_data['average_mae'] = IQA_metrics_data['accumulate_mae'] / IQA_metrics_data['accumulate_number_of_input_samples_processed']
	
	# LPIPS part
	batch_average_lpips = lpips(enhanced_image, img_lowlight).item()
	IQA_metrics_data['accumulate_lpips'] += batch_average_lpips * sample_size
	IQA_metrics_data['average_lpips'] = IQA_metrics_data['accumulate_lpips'] / IQA_metrics_data['accumulate_number_of_input_samples_processed']

	# Summary of the important IQA metric results
	print('Accumulated processed sample numbers:%d, Average PSNR: %.4f dB, Average SSIM: %.4f, Average MAE: %.4f, Average LPIPS: %.4f' % (
                        IQA_metrics_data['accumulate_number_of_input_samples_processed'], IQA_metrics_data['average_psnr'], IQA_metrics_data['average_ssim'], IQA_metrics_data['average_mae'], IQA_metrics_data['average_lpips']))
	
def ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data): # We define ComputationalComplexity_metrics_data as the input argument, so that the results related to ComputationalComplexity_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	
	# Calculate the average runtime using the total batch duration and total number of input samples (means the average runtime for an input sample)
	ComputationalComplexity_metrics_data['average_runtime'] = ComputationalComplexity_metrics_data['accumulate_batch_duration'] / IQA_metrics_data['accumulate_number_of_input_samples_processed']
	
	# calflops part: To calculate MACs, FLOPs, and trainable parameters
	model_name = config.model_name # The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
	batch_size = 1 # reference batch size for calflops
	reference_input_shape = (batch_size, img_lowlight.shape[1], img_lowlight.shape[2], img_lowlight.shape[3]) # (reference batch size for calflops, channel numbers of each input sample, height dimension of each input sample, width dimension of each input sample)
	ComputationalComplexity_metrics_data['FLOPs'], ComputationalComplexity_metrics_data['MACs'], ComputationalComplexity_metrics_data['trainable_parameters'] = calculate_flops(model=DCE_net,  # The model created
																																								input_shape=reference_input_shape,
																																								output_as_string=True,
																																								output_precision=4)
	
	# Summary of the computational complexity metric results
	print("Model:%s,	FLOPs:%s,   MACs:%s,   Trainable parameters:%s , Average runtime:%.4f second(s)\n" %(model_name, ComputationalComplexity_metrics_data['FLOPs'], ComputationalComplexity_metrics_data['MACs'], ComputationalComplexity_metrics_data['trainable_parameters'], ComputationalComplexity_metrics_data['average_runtime']))

	return reference_input_shape



# **Part 1: ArgumentParser, usable only if you run this script**
if __name__ == '__main__':

	parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
	# Input Parameters
	parser.add_argument('--model_name', type=str, default= "Zero-DCE") # The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
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

# **Part 2: Select the device for computations**
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# **Part 3: Initialize the functions to calculate performance metrics**
	ssim = SSIM(data_range=1.).to(device) # Performance metric: SSIM
	psnr = PSNR(data_range=1.).to(device) # Performance metric: PSNR
	mae = MAE(num_outputs=1).to(device) # Performance metric: MAE
	lpips = LPIPS(net_type='alex', normalize=True).to(device) # Performance metric: LPIPS (perceptual loss provided by a pretrained learning-based model), using alexnet as the backbone (LPIPS with alexnet performs the best according to its official Github)

# **Part 4: Initialize a dictionary to store the Image Quality Assessment (IQA) metric data , so they will be updated from time to time later**
	IQA_metrics_data = {'iteration':0, 'average_psnr': 0, 'average_ssim': 0, 'average_mae': 0, 'average_lpips': 0, 'accumulate_number_of_input_samples_processed': 0, 'batch_duration':0, 'accumulate_psnr': 0, 'accumulate_ssim': 0, 'accumulate_mae': 0,'accumulate_lpips': 0}

# **Part 5: Initialize a dictionary to store the computational complexity metric data , so they will be updated from time to time later**
	ComputationalComplexity_metrics_data ={'average_runtime':0, 'trainable_parameters':0, 'MACs':0, 'FLOPs': 0, 'accumulate_batch_duration':0}

	with torch.no_grad():
		
		# **Part 6: Initialize the model**
		DCE_net = ZTWV_model.enhance_net_nopool().to(device) # Move the model to cuda so that the samples in each batch can be processed simultaneously
		DCE_net.eval() # Set the model to evaluation mode

		if config.load_pretrain == True: # When NOT set to train mode 
			DCE_net.load_state_dict(torch.load(config.pretrain_dir)) # load the parameters (weights & biases) of the model obtained/learned at the snapshot (EG: at a particular epoch)

		# **Part 7: Prepare dataset (From loading images, preprocessing image data, to converting image data into tensor with dimension sequence reordered if required)**
		testPath =  config.lowlight_images_test_path # The absolute path that stores the test data
		test_list = os.listdir(testPath) # Returns a list containing the names of the entries in a directory specified by path. Here, the returned list is ['DICM', 'LIME'], the names of subfolder inside the folder "test_data". 
		
		for testfile_name in test_list: # For each subfolder inside the test_data folder (DCIM subfolder first, then only LIME subfolder)
			if testfile_name == "DICM": # Only takes the DICM test data available in testPath
				test_path_withFolderName = testPath + testfile_name +"/"
				print("test_path_withFolderName:", test_path_withFolderName)
				test_dataset = ZTWV_dataloader.lowlight_loader(test_path_withFolderName) # Create custom test dataset: Take the images available in test_path, then preprocess them, and convert them into a tensor
		
		# **Part 8: Define the paths/folders that store the results obtained from this script**
		resultPath = testPath.replace('test_data','result') # Create a new path by changing a part of resultPath
		resultPath_subfolder = "self_CompileMetrics" # Define the folder that stores the enhanced images and the grids of input-output image pairs involved in metric calculations as different folder respectively
			
			# For the folder that stores the enhanced images
		resultPath_subfolder_individual = "self_CompileMetrics_EnhancedImages" # Define the folder that stores the enhanced images
		resultPath_EnhancedResults = os.path.join(resultPath, resultPath_subfolder, resultPath_subfolder_individual) # Define the absolute path to the folder that stores the enhanced images
		os.makedirs(resultPath_EnhancedResults, exist_ok=True) # Ensure the path that stores the enhanced images is created
			
			# For the folder that stores the grids of input-output image pairs involved in metric calculations
		resultPath_subfolder_imagepairs = "self_CompileMetrics_ImagePairs" # Define the folder that stores the grids of input-output image pairs involved in metric calculations
		resultPath_ImagePairsResults = os.path.join(resultPath, resultPath_subfolder, resultPath_subfolder_imagepairs) # Define the absolute path to the folder that stores the grids of input-output image pairs involved in metric calculations
		os.makedirs(resultPath_ImagePairsResults, exist_ok=True) # Ensure the path that stores the grids of input-output image pairs involved in metric calculations is created
		
			# For the csv that stores the metrics data
		resultPath_subfolder_csv = "self_CompileMetrics_csvFile" # Define the folder that stores the csv file
		resultPath_csv = os.path.join(resultPath, resultPath_subfolder, resultPath_subfolder_csv) # Define the absolute path to the folder that stores the csv file
		os.makedirs(resultPath_csv, exist_ok=True) # Ensure the path that stores the csv file is created
		current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
		csv_result_filename = 'LLIE-FinalResults-summary-' + config.model_name + '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
		csv_result_filepath = os.path.join(resultPath,resultPath_subfolder,resultPath_subfolder_csv,csv_result_filename) # Create the path to the csv that stores the metrics data
		with open(csv_result_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
			writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
			writer.writeheader() # The writer writes the header on the csv file


		# **Part 9: Split the dataset into batches of samples**
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True) # verified that the indices used in each epoch are reproducible at every time runing the script
		
		test_bar = tqdm(test_loader) # tqdm automatically detects the length of the iterable object (train_loader) [The length of the iterable object = The total number of batches of samples] and generates a progress bar that updates dynamically as each item=(batch of samples) is processed.
		torch.manual_seed(1) # Initialize/seed the PyTorch random number generator (RNG) state, so that the PyTorch RNG will generate the reproducible/deterministic base seed whenever the dataloader object is called (If the PyTorch RNG is initialized/seeded only once before the dataloader object is called the first time, it will generate different base seeds at each time the dataloader object is called, but the base seeds are reproducible/deterministic. [EG: If the PyTorch RNG is initialized/seeded only once before the dataloader object is called the first time (before the first epoch), the sequence of shuffled indices used in each epoch will be different but are reproducible/deterministic. But if the PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch), the sequence of shuffled indices used in each epoch will be the same and are reproducible/determistic.]). Thus, the sequence of shuffled indices used in each epoch are reproducible/determistic at every time runing the script. This is because for the case of multiprocessing/multiworkers (such that num_workers>0), each worker will have its PyTorch seed set to base_seed + worker_id, where base_seed is generated by PyTorch RNG based on its state (its state will change everytime it is called, but its state can be initialized with torch.manual_seed()) while each worker has unique fixed worker id. 
		# Validation starts here
		for iteration, img_lowlight in enumerate(test_bar):
			
			IQA_metrics_data['iteration'] = iteration # Update the current iteration (The number of epoch and batch, when necessary) to the IQA_metrics_data dictionary
			
			# **Part 10: Get a batch of samples at each iteration**
			img_lowlight = img_lowlight.to(device) # Since this batch of train_batch_size samples are moved to cuda for processinng, they will be processed simultaneously.
			
			# **Part 11: Perform image enhancement on the current batch of samples using the model**
			enhanced_image = image_enhancement(IQA_metrics_data, ComputationalComplexity_metrics_data) # image_enhancement() returns enhanced_image. enhanced_image will then be used to calculate IQA metrics in IQA_metrics_calculation() and for object detection tasks at later phase.
			
			# **Part 12: Perform Image Quality Assessment (IQA) metric calculations**
			IQA_metrics_calculation(IQA_metrics_data) 

			# **Part 13: Record the calculated (IQA) metrics to that csv file**
			with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
				writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
				writer.writerow(IQA_metrics_data) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file
				
			# **Part 14: Save the enhanced images and grid of input-output image pairs involved in the current iteration**
			save_image()  

			
		# **Part 15: Perform computation complexity metric calculations**
		reference_input_shape = ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data)

		# **Part 16: Record the calculated computation complexity metrics to that csv file**
		with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of ComputationalComplexity_metrics_data dictionary to that csv file.
			writer = csv.DictWriter(csvfile, fieldnames=ComputationalComplexity_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
			writer.writeheader() # The writer writes the header on the csv file
			writer.writerow(ComputationalComplexity_metrics_data)  # The writer writes the data (value) of ComputationalComplexity_metrics_data dictionary in sequence as a row on the csv file

		# **Part 17: Show the final results of the model after completing the training/inference**
		print('\n----------------------------------Final results of the LLIE model [%s] performance----------------------------------' %(config.model_name))
		print('\nA)----IQA metrics summary----\n1) Average PSNR: %.4f dB\n2) Average SSIM: %.4f\n3) Average MAE: %.4f\n4) Average LPIPS: %.4f\n**[Note: Each result above is calculated for per output sample (enhanced image), which averaged on %d output samples of (channel=%d, height=%d, width=%d)]**\n\nB)----Computational complexity metrics summary----\n1) Average runtime: %f second(s)\n2) Trainable parameters: %s\n3) MACs: %s\n4) FLOPs: %s\n**[Note:\n# Average runtime is calculated for per input sample, which averaged on %d input samples of (channel=%d, height=%d, width=%d)\n# MACs and FLOPs are respectively calculated using the reference input shape: (batch_size=%d, channel=%d, height=%d, width=%d]**\n\n' % (
                        IQA_metrics_data['average_psnr'], IQA_metrics_data['average_ssim'], IQA_metrics_data['average_mae'], IQA_metrics_data['average_lpips'], 
						IQA_metrics_data['accumulate_number_of_input_samples_processed'], reference_input_shape[1], reference_input_shape[2], reference_input_shape[3],
						ComputationalComplexity_metrics_data['average_runtime'], ComputationalComplexity_metrics_data['trainable_parameters'], ComputationalComplexity_metrics_data['MACs'], ComputationalComplexity_metrics_data['FLOPs'],
						IQA_metrics_data['accumulate_number_of_input_samples_processed'], reference_input_shape[1], reference_input_shape[2], reference_input_shape[3], 
						reference_input_shape[0], reference_input_shape[1], reference_input_shape[2], reference_input_shape[3]))


# Notes: 
# 1) Idea: 
# A) We calculate the metrics by using the enhanced image before being saved because after saving the enhanced image, the pixel values will change slightly. We do this also equivalently build the pipiline of computer vision task with LLIE algorithm, such that the enhanced image is directly passed to the following modules for the relevant application (EG: object detection) without the enhanced image being saved on that device first.
# B) We use calflops to calculate the MACs (forward MACs only), FLOPs (forward FLOPs only), and Parameter using the batch size of 1, to approximately show the computational complexity (EG: Resource efficiency [computational cost], memory efficiency, and energy consumption) required by the model for every single input sample. This is because the MACs and FLOPs are dependent/scalable with the batch size, such that we can easily calculate (MACs for batch size of X) = (X)*(MACs for batch size of 1) and (FLOPs for batch size of X) = (X)*(FLOPs for batch size of 1) if required. But the Parameter is independent on the batch size. Take note the unit of FLOPs shown by calflops at the end has wrong unit (should be GFLOPs instead of GFLOPS), and this can be verified with the value provided by other publicly available FLOPs counters (verified that other publicly available FLOPs counters give the similar FLOPs value compared to the one of calflops). 
# C) We calculate the average runtime using the batch size that yields the optimum performance/results of my proposed model, such that (average runtime) = (Summation of runtime of each batch of each epoch)/(Total number of samples processed by the model), to approximately show the average duration required by the model to process every single input sample. This is because average runtime could be dependent/scalable with the batch size. The runtime of each batch of each epoch is defined as the duration starting from the model takes a batch of input samples and ending with the model provides their corresponding output samples (which is the enhanced images), such that (runtime of each batch of each epoch)=(The time the model provides output samples)-(The time the model takes input samples).
# 
# 2) Done:
# A) Create a randomly shuffled dataset using the samples/images from the given dataset path, after resizing them into a square images (performed by lowlight_loader() in dataloader.py). 
# B) Calculate the IQA metrics using the input-output image pairs, where the output images involved in metric calculations are the ones right after provided by the model and before being saved as image files on the device.
# C) Calculate the average runtime based on the total number of input samples, while calculate the trainable parameters, MACs, and FLOPs based on the reference input shape of (batch size = 1, channel numbers of each input sample, height dimension of each input sample, width dimension of each input sample).
# D) The enhanced images provided by the model will be saved as image files on the device, with the filename according to the sequence being processed by the model.
# E) The input-output image pairs involved in metric calculations at each batch will be organized as a grid for comparison and saved as an image file, with the filename according to the batch number.
# F) Record the IQA metric data at each epoch & batch and computational complexity of the model on a single csv file.
# G) The final results of the LLIE model performance will be printed at last.
# H) Study why when the script is run multiple times, only the Average PSNR fluctuates (compare by using saved enhanced image vs the enhanced image before being saved) "Answer: Because the state of PyTorch RNG is not fixed so that the series of shuffled indices is not reproducible and deterministic"
# I) Study why initially keep mentioning "Total training examples", but this is not important. "Answer: Because [print("Total training examples:", len(self.train_list))] is defined in the __init__ of the dataset object called lowlight_loader in dataloader.py, which will be called when the dataset object is initialized. "


