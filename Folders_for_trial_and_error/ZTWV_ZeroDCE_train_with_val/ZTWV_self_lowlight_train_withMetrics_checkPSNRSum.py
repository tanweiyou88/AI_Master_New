import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import time
import ZTWV_dataloader_WithPair
import ZTWV_model
import ZTWV_Myloss
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import numpy as np

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE
from tqdm import tqdm
import torch.utils.data

from calflops import calculate_flops # for calflops to calculate MACs, FLOPs, and trainable parameters
import csv



def load_data(config):
	train_dataset = ZTWV_dataloader_WithPair.LLIEDataset(config.train_GroundTruth_root, config.train_Input_root)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
												num_workers=config.num_workers, pin_memory=True)
	val_dataset = ZTWV_dataloader_WithPair.LLIEDataset(config.val_GroundTruth_root, config.val_Input_root)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
												num_workers=config.num_workers, pin_memory=True)

	return train_loader, len(train_loader), val_loader, len(val_loader)

# function used to initialize the parameters (weights and biases) of the network. This function will be used only when the network is not using the pretrained parameters
def weights_init(m): # A custom function that checks the layer type and applies an appropriate initialization strategy for each case, so that it ensures that the initialization is applied consistently across all layers. It iterating through model layers and systematically apply weight initialization across all layers in a model using the model.apply method. Means in this case, all layers whose name containing "Conv" and "BatchNorm" as part will be initialized using the defined methods.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # if layer whose part of its name having 'Conv' is not found by find(), it will return '-1'
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: # if layer whose part of its name having 'BatchNorm' is not found by find(), it will return '-1'
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# def save_image():
# 	# Save the enhanced images involved in the current iteration 
# 	for count, image in enumerate(enhanced_image):
# 		sample_size = enhanced_image.shape[0] # Get the total number of enhanced_image samples in current iteration, can be equivalent to batch size
# 		# Save each individual enhanced image, according to the sequence being processed by the model
# 		torchvision.utils.save_image(
# 			image,
# 			os.path.join(resultPath_EnhancedResults, 'enhanced_image_{}.jpg'.format( IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] - sample_size + count + 1) )
# 		)

# 	# Create a grid of input-output image pairs involved in the current iteration 
# 	grid_image = torchvision.utils.make_grid(torch.cat((img_lowlight, enhanced_image), 0), nrow=img_lowlight.shape[0])
	
# 	# Save the created grid of input-output image pairs involved in the current iteration 
# 	torchvision.utils.save_image(
# 		grid_image,
# 		os.path.join(resultPath_ImagePairsResults, 'InputEnhanced_ImagePair_{}.jpg'.format(iteration + 1))
# 	)
	
def IQA_metrics_calculation(IQA_metrics_data): # We define IQA_metrics_data as the input argument, so that the results related to IQA_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	sample_size = LL_image.shape[0] # Get the total number of input samples (img_lowlight.shape[0]) that have been processed by the model in the current iteration, can be equivalent to batch size.
	IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] += sample_size # Update the processed input samples in the current iteration. So that at the end, IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] stores the total number of input samples that have been processed by the model.

	# PSNR part [Focus here]

	# # Done, Use this concept to calculate PSNR: Implement PNSR_averaged_perEpoch = [PNSR_averaged_batch1+PNSR_averaged_batch2+...]/[Total number of batches]; PNSR_averaged_batchN=[PNSR_image1+PNSR_image2+...]/[Total number of images in that batch] (where PNSR_imageN is obtained using the sum_squared_error=[Summation of differences of each pixel on the image itself, compared with its GroundTruth image] and num_obs=[Total number of image pixels on the image itself])
	# batch_accumulate_psnr = 0.
	# for index in range(len(enhanced_image)):
	# 	print('index:', index)
	# 	extracted_single_enhanced_image = enhanced_image[index]
	# 	# print('extracted_single_enhanced_image:', extracted_single_enhanced_image)
	# 	extracted_single_ori_image = ori_image[index]
	# 	# print('extracted_single_ori_image:', extracted_single_ori_image)
	# 	psnr_single_enhanced_ori_ImagePair = psnr(extracted_single_enhanced_image, extracted_single_ori_image).item()
	# 	print('psnr_single_enhanced_ori_ImagePair:', psnr_single_enhanced_ori_ImagePair)
	# 	batch_accumulate_psnr += psnr_single_enhanced_ori_ImagePair
	# 	print('batch_accumulate_psnr:', batch_accumulate_psnr)

	# # batch_average_psnr = psnr(enhanced_image, ori_image).item() # torchmetrics.PeakSignalNoiseRatio first calculates the PSNR of each image pair in a batch of images, then return the average PSNR of all image pairs in that batch. The mathematical concept behinds is: (average PSNR)=([Summation of PSNR of all image pairs in a batch of images]/[Total number of image pairs in that batch of images, which is same as the batch size]) of that batch of images.
	# # print('psnr(enhanced_image, ori_image):', psnr(enhanced_image, ori_image))
	# IQA_metrics_data['epoch_accumulate_psnr'] += batch_accumulate_psnr
	# print('epoch_accumulate_psnr:', IQA_metrics_data['epoch_accumulate_psnr'])
	# IQA_metrics_data['average_psnr'] = IQA_metrics_data['epoch_accumulate_psnr'] / IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] # Get the average PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (Summation of PSNR of all image pairs the model has gone through)/[Total batches of images the model has gone through, which is same as the total number of image pairs the model has gone through]).


	# Original, Ignore this: It might use wrong concept to calculate average PSNR of multiple images. For each batch, it first calculates the MSE averaged on the total number of image pixels in a batch first, then only takes that MSE to calculate PSNR as the PSNR_averaged_batch. To illustrates, it uses MSE_averaged_from_images_in_each_batch = [Summation of differences of each pixel in the images of a batch]/[Total number of image pixels in that batch], instead of PNSR_averaged_batchN = [PNSR_image1+PNSR_image2+...]/[Total number of images in that batch]. The correct way to calculate is using PNSR_averaged_batchN=[PNSR_image1+PNSR_image2+...]/[Total number of images in that batch] (where PNSR_imageN is obtained using the sum_squared_error=[Summation of differences of each pixel on the image itself, compared with its GroundTruth image] and num_obs=[Total number of image pixels on the image itself]) to get PNSR_averaged_perEpoch = [PNSR_averaged_batch1+PNSR_averaged_batch2+...]/[Total number of batches].
	batch_average_psnr = psnr(enhanced_image, ori_image).item() # torchmetrics.PeakSignalNoiseRatio first calculates the PSNR of each image pair in a batch of images, then return the average PSNR of all image pairs in that batch. The mathematical concept behinds is: (average PSNR)=([Summation of PSNR of all image pairs in a batch of images]/[Total number of image pairs in that batch of images, which is same as the batch size]) of that batch of images.
	# print('psnr(enhanced_image, ori_image):', psnr(enhanced_image, ori_image))
	IQA_metrics_data['accumulate_psnr'] += batch_average_psnr * sample_size # Get the total PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (batch_psnr * cfg.batch_size)=(Summation of PSNR of all image pairs in a batch of images).
	IQA_metrics_data['average_psnr'] = IQA_metrics_data['accumulate_psnr'] / IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] # Get the average PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (Summation of PSNR of all image pairs the model has gone through)/[Total batches of images the model has gone through, which is same as the total number of image pairs the model has gone through]).

	# SSIM part
	batch_average_ssim = ssim(enhanced_image, ori_image).item()
	IQA_metrics_data['accumulate_ssim'] += batch_average_ssim * sample_size
	IQA_metrics_data['average_ssim'] = IQA_metrics_data['accumulate_ssim'] / IQA_metrics_data['accumulate_number_of_val_input_samples_processed']

	# MAE part
	batch_average_mae = mae(enhanced_image, ori_image).item()
	IQA_metrics_data['accumulate_mae'] += batch_average_mae * sample_size
	IQA_metrics_data['average_mae'] = IQA_metrics_data['accumulate_mae'] / IQA_metrics_data['accumulate_number_of_val_input_samples_processed']
	
	# LPIPS part
	batch_average_lpips = lpips(enhanced_image, ori_image).item()
	IQA_metrics_data['accumulate_lpips'] += batch_average_lpips * sample_size
	IQA_metrics_data['average_lpips'] = IQA_metrics_data['accumulate_lpips'] / IQA_metrics_data['accumulate_number_of_val_input_samples_processed']

def ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data): # We define ComputationalComplexity_metrics_data as the input argument, so that the results related to ComputationalComplexity_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	
	# Calculate the average runtime using the total batch duration and total number of input samples (means the average runtime for an input sample)
	ComputationalComplexity_metrics_data['average_val_runtime'] = ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteTraining'] / (IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] * config.num_epochs)
	
	# calflops part: To calculate MACs, FLOPs, and trainable parameters
	# model_name = config.model_name # The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
	batch_size = 1 # reference batch size for calflops
	reference_input_shape = (batch_size, LL_image.shape[1], LL_image.shape[2], LL_image.shape[3]) # (reference batch size for calflops, channel numbers of each input sample, height dimension of each input sample, width dimension of each input sample)
	ComputationalComplexity_metrics_data['FLOPs'], ComputationalComplexity_metrics_data['MACs'], ComputationalComplexity_metrics_data['trainable_parameters'] = calculate_flops(model=DCE_net,  # The model created
																																								input_shape=reference_input_shape,
																																								print_results=False,
                    																																			print_detailed=False,
																																								output_as_string=False,
																																								output_precision=4)

	return reference_input_shape

def save_model(epoch, path, net, optimizer, net_name): # this function saves model checkpoints
    if not os.path.exists(os.path.join(path, net_name)): # if the folder that stores the checkpoints does not exist
        os.makedirs(os.path.join(path, net_name))
	
	# torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') # The format used by the author. Save the parameters (weigths and biases) of the model at the specified snapshot (EG:epoch) as a pth file	
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, # save the checkpoints
               f=os.path.join(path, net_name, '{}_{}_{}_{}_checkpoint.pt'.format(config.model_name, config.dataset_name, 'Epoch', epoch)))


# **Part 1: ArgumentParser, usable only if you run this script**
if __name__ == '__main__':

	parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
	# Input Parameters
	parser.add_argument('--model_name', type=str, default= "Zero-DCE") # The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
	parser.add_argument('--dataset_name', type=str, default= "dataset1")
	parser.add_argument('--train_GroundTruth_root', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/dataset/train/GroundTruth") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--train_Input_root', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/dataset/train/Input") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--val_GroundTruth_root', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/dataset/val/GroundTruth") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--val_Input_root', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/dataset/val/Input") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--lr', type=float, default=0.0001) # Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=2)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--val_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	# parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=1)
	parser.add_argument('--snapshots_folder', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	# parser.add_argument('--pretrain_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/snapshots/Epoch99.pth") # The pretrained model parameters provided by the author
	parser.add_argument('--checkpoint_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/results/model_parameters/Zero-DCE/enhance_1.pt") # The pretrained model parameters obtained by myself
	parser.add_argument('--BatchesNum_ImageGroups_VisualizationSample', type=int, default=20) # At each epoch, save the maximum first N batches of LL_image (Input), enhanced_image (Output), and ori_image (GroundTruth) image groups in validation as the samples to visualize the network performance 
	
	# The parse_args() object (config=self) takes the data (values) you provide to your positional/optional arguments on command line interface or within the () of parse_args(), then converts them into the required data type as mentioned in add_argument() respectively. 
	# So you can access the data of a positional/optional argument by using the syntax args.argument_name (EG: config.lowlight_images_path).
	config = parser.parse_args() 

### **Part 2: Initialization**


# **Subpart 1: Select the device for computations**
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	torch.manual_seed(1) # This function ensures the reproducibility of the parameters (weights and biases) of the network which are initialized using random normal functions. The PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch)

# **Initialize folders
	# **Subpart 5: Define the paths/folders that store the results obtained from this script**
	resultPath = config.train_GroundTruth_root.replace('dataset/train/GroundTruth','results') # Create a new path by changing a part of resultPath
	# resultPath_subfolder = "self_CompileMetrics" # Define the folder that stores the enhanced images and the grids of input-output image pairs involved in metric calculations as different folder respectively
		
	# # For the folder that stores the enhanced images
	# resultPath_subfolder_individual = "EnhancedImages" # Define the folder that stores the enhanced images
	# resultPath_EnhancedResults = os.path.join(resultPath, resultPath_subfolder_individual) # Define the absolute path to the folder that stores the enhanced images
	# os.makedirs(resultPath_EnhancedResults, exist_ok=True) # Ensure the path that stores the enhanced images is created
		
	# # For the folder that stores the grids of input-output image pairs involved in metric calculations
	# resultPath_subfolder_imagepairs = "ImagePairs" # Define the folder that stores the grids of input-output image pairs involved in metric calculations
	# resultPath_ImagePairsResults = os.path.join(resultPath, resultPath_subfolder_imagepairs) # Define the absolute path to the folder that stores the grids of input-output image pairs involved in metric calculations
	# os.makedirs(resultPath_ImagePairsResults, exist_ok=True) # Ensure the path that stores the grids of input-output image pairs involved in metric calculations is created
	

	# For the folder that stores the grids of input-output image pairs involved in metric calculations
	resultPath_subfolder_modelparameters = "model_parameters" # Define the folder that stores the grids of input-output image pairs involved in metric calculations
	resultPath_ModelParametersResults = os.path.join(resultPath, resultPath_subfolder_modelparameters) # Define the absolute path to the folder that stores the grids of input-output image pairs involved in metric calculations
	os.makedirs(resultPath_ModelParametersResults, exist_ok=True) # Ensure the path that stores the grids of input-output image pairs involved in metric calculations is created
	

	# For the csv that stores the metrics data
	resultPath_subfolder_csv = "csvFile" # Define the folder that stores the csv file
	resultPath_csv = os.path.join(resultPath, resultPath_subfolder_csv) # Define the absolute path to the folder that stores the csv file
	os.makedirs(resultPath_csv, exist_ok=True) # Ensure the path that stores the csv file is created
	current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
	csv_result_filename = 'LLIE-ValidationResults-History-' + config.model_name + '-' + config.dataset_name + '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
	csv_result_filepath = os.path.join(resultPath_csv, csv_result_filename) # Create the path to the csv that stores the metrics data
	# with open(csv_result_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
	# 	writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
	# 	writer.writeheader() # The writer writes the header on the csv file

	sample_output_folder = os.path.join(resultPath,'sample_output_folder') # Create a new path by changing a part of resultPath
	sample_dir = os.path.join(sample_output_folder, config.model_name)
	if not os.path.isdir(sample_dir):
		os.makedirs(sample_dir)

# **Subpart 2: Initialize training and validation dataset**
	train_loader, train_number, val_loader, val_number = load_data(config)

	
	# # **Check index reproducibility**, NOT yet checked
	# Indices_history = []
	# # for iteration, (indices, img_lowlight) in enumerate(test_bar):
	# for i in range(3):
	# 	print(f"\nEpoch {i}:")
	# 	# torch.manual_seed(1) # the PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch)
	# 	for iteration, (indices, img_lowlight) in enumerate(train_loader):
	# 		print(f"Batch {iteration}, Indices: {indices}")
	# 		indices_cpu = indices.cpu()
	# 		indices_cpu_numpy = indices_cpu.numpy() #convert to Numpy array
	# 		Indices_history.append(indices_cpu_numpy)

	# 	csv_IndicesHistory_filename = "Zero-DCE-IndicesHistory"+ '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
	# 	csv_IndicesHistory_filepath = os.path.join(resultPath,resultPath_subfolder,resultPath_subfolder_csv,csv_IndicesHistory_filename) # Create the path to the csv that stores the metrics data
	# 	df = pd.DataFrame(Indices_history) #convert the numpy array into a dataframe
	# 	df.to_csv(csv_IndicesHistory_filepath,index=False) #save the dataframe into a CSV file


# **Subpart 3: Initialize the network**
	DCE_net = ZTWV_model.enhance_net_nopool().cuda() # Move the model to the specified device so that the samples in each batch can be processed simultaneously

	DCE_net.apply(weights_init)
	if config.load_pretrain == True: # When using the pretrained network parameters
		# DCE_net.load_state_dict(torch.load(config.pretrain_dir)) # load the pretrained parameters (weights & biases) of the model obtained/learned at the snapshot (EG: at a particular epoch)
		checkpoint = torch.load(config.checkpoint_dir)
		DCE_net.load_state_dict(checkpoint['model_state_dict']) # load the parameters (weights & biases) of the model obtained/learned at a particular checkpoint (EG: at a particular epoch)


# **Subpart 4: Initialize the loss functions used for training**
	L_color = ZTWV_Myloss.L_color()
	L_spa = ZTWV_Myloss.L_spa()
	L_exp = ZTWV_Myloss.L_exp(16,0.6)
	L_TV = ZTWV_Myloss.L_TV()

# **Subpart 5: Initialize the optimizer used for training**
	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay) # Set the hyperparameters for the optimizer to adjust the model parameters (weights and biases), after each loss.backward() [perform backpropagation = calculate the partial derivative of loss with respect to each weight] of a batch of samples

# **Subpart 6: Initialize the IQA metric-related functions to calculate IQA results during validation**
	ssim = SSIM(data_range=1.).cuda() # Performance metric: SSIM
	psnr = PSNR(data_range=1., reduction='sum').cuda()# Performance metric: PSNR
	mae = MAE(num_outputs=1).cuda() # Performance metric: MAE
	lpips = LPIPS(net_type='alex', normalize=True).cuda() # Performance metric: LPIPS (perceptual loss provided by a pretrained learning-based model), using alexnet as the backbone (LPIPS with alexnet performs the best according to its official Github)


### **Part 3: Training**
	print('Start train')
	DCE_net.train()

	# **Subpart 2: Initialize a dictionary to store the computational complexity metric data , so they will be updated from time to time later**
	ComputationalComplexity_metrics_data ={'average_val_runtime':0, 'trainable_parameters':0, 'MACs':0, 'FLOPs': 0, 'accumulate_val_batch_duration_forCompleteTraining':0}
	
	torch.manual_seed(1) # This function ensures the reproducibility of the shuffled indices used in the train_loader and val_loader. The PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch)
	for epoch in range(config.num_epochs):

		# **Subpart 1: Initialize a dictionary to store the Image Quality Assessment (IQA) metric data , so they will be updated from time to time later**
		IQA_metrics_data = {'epoch':0, 'average_psnr': 0, 'average_ssim': 0, 'average_mae': 0, 'average_lpips': 0, 'accumulate_number_of_val_input_samples_processed': 0, 'accumulate_psnr': 0, 'accumulate_ssim': 0, 'accumulate_mae': 0,'accumulate_lpips': 0}
		IQA_metrics_data['epoch'] = epoch # Update the current epoch (The number of epoch and batch, when necessary) to the IQA_metrics_data dictionary
		
		total_loss = 0.

		train_bar = tqdm(train_loader) # tqdm automatically detects the length of the iterable object (train_loader) [The length of the iterable object = The total number of batches of samples] and generates a progress bar that updates dynamically as each item=(batch of samples) is processed.
		for iteration, (ori_image, LL_image) in enumerate(train_bar):
			# count = epoch * train_number + (iteration + 1)
			ori_image, LL_image = ori_image.cuda(), LL_image.cuda() # The model requires normallight-lowlight image pairs as inputs          
			# print('ori_image:', ori_image)
			# print('LL_image:', LL_image)
			enhanced_image_1,enhanced_image,A = DCE_net(LL_image) # Call the model to enhance this batch of samples, by returning each image enhanced at 4th enhancing iteration, each image enhanced at last enhancing iteration, and the 24 curve parameter maps used to enhance this batch of samples at different stages respectively.
				
			Loss_TV = 200*L_TV(A) # Calculate the illumination smoothness loss of this batch of samples
			loss_spa = torch.mean(L_spa(enhanced_image, LL_image)) # Calculate the spatial consistency loss of this batch of samples
			loss_col = 5*torch.mean(L_color(enhanced_image)) # Calculate the color constancy loss of this batch of samples
			loss_exp = 10*torch.mean(L_exp(enhanced_image)) # Calculate the exposure control loss of this batch of samples
			# Total loss of this batch of samples (best_loss, considering the weightage of each loss contributed to the total loss)
			loss =  Loss_TV + loss_spa + loss_col + loss_exp 
			total_loss = total_loss + loss.item() # total_loss stores the total loss in an epoch, for record purpose only (not for backpropagation)

			optimizer.zero_grad() # Remove all previously calculated partial derivatives
			loss.backward() # Perform backpropagation
			torch.nn.utils.clip_grad_norm_(DCE_net.parameters(),config.grad_clip_norm) # Perform Gradient Clipping by Norm to prevent the gradients from becoming excessively large during the training of neural networks, which will lead to exploding gradients problem.
			optimizer.step() # Update the parameters (weights and biases) of the model

			# IMPORTANT PART: Contains meaningful information; END HERE
			# summary.add_scalar('loss', loss.item(), count)
			# summary.add_scalar('recon_loss', recon_loss.item(), count)
			train_bar.set_description_str('Epoch: {}/{} | Iteration: {}/{} | lr: {:.6f} | Average total loss: {:.6f}; Loss_TV: {:.6f}; loss_spa: {:.6f}; loss_col: {:.6f}; loss_exp: {:.6f}'
					.format(IQA_metrics_data['epoch'] + 1, config.num_epochs, iteration + 1, train_number,
							optimizer.param_groups[0]['lr'], 
							total_loss/(iteration+1), Loss_TV.item(), loss_spa.item(), loss_col.item(), loss_exp.item(),
						)
					)
			
		# scheduler.step()

		# save model parameters at certain checkpoints
		if ((iteration+1) % config.snapshot_iter) == 0: # at every (snapshot_iter)th iteration
			save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, config.model_name)	

		# -------------------------------------------------------------------
		### **Part 4: Validation**    

		print('Epoch: {}/{} | Validation Model Saving Images'.format(IQA_metrics_data['epoch'] + 1, config.num_epochs))
        
		DCE_net.eval()

		val_bar = tqdm(val_loader)

		
		max_firstNBatches_ImageGroups_VisualizationSample = config.BatchesNum_ImageGroups_VisualizationSample
		if len(val_loader) < max_firstNBatches_ImageGroups_VisualizationSample: # len(val_loader) returns the number of batches of the dataset = total number of iterations/batches. len(val_loader.dataset) returns the number samples in the dataset = the number of input-output-GroundTruth image groups.
			max_firstNBatches_ImageGroups_VisualizationSample = len(val_loader) # when the total batches number of the dataset is less than the specified maximum batches number for visualization, takes the total batches number of the dataset as the maximum batches number for visualization

		save_image = None

		for iteration, (ori_image, LL_image) in enumerate(val_bar):
            
			ori_image, LL_image = ori_image.cuda(), LL_image.cuda()

			with torch.no_grad():
				# **Subpart 8: Perform image enhancement on the current batch of samples using the model**
				start = time.time() # Get the starting time of image enhancement process on a batch of samples, in the unit of second.

				_,enhanced_image,_ = DCE_net(LL_image) # Perform image enhancement using Zero-DCE on the current batch of images/samples. enhanced_image stores the enhanced images.

				val_batch_duration = (time.time() - start) # Get the duration of image enhancement process on the current batch of samples
				# IQA_metrics_data['batch_duration'] = batch_duration # Update the current batch duration to the IQA_metrics_data dictionary
				# ComputationalComplexity_metrics_data['accumulate_batch_duration'] += IQA_metrics_data['batch_duration'] # Update the accumulate batch duration to the ComputationalComplexity_metrics_data dictionary
				ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteTraining'] += val_batch_duration # Update the accumulate batch duration to the ComputationalComplexity_metrics_data dictionary
				# print("\nDuration of image enhancement process on the current batch of input samples [second (s)]:", IQA_metrics_data['batch_duration'])
			
			# **Subpart 9: Perform Image Quality Assessment (IQA) metric calculations**
			IQA_metrics_calculation(IQA_metrics_data) 

			if iteration <= (max_firstNBatches_ImageGroups_VisualizationSample - 1):   # before reaching the first max_step number of iterations/batches in each epoch, the LL_image (Input), enhanced_image (Output), and ori_image (GroundTruth) image groups the network has dealt with will be concatenated horizontally together as an image grid. So max_step determines the number of groups will be concatenated horizontally in the image grid. In other words, only the first max_step groups will be chosen as the samples to be concatenated as the image grid to show/visualize the network performance.
				sv_im = torchvision.utils.make_grid(torch.cat((LL_image, enhanced_image, ori_image), 0), nrow=ori_image.shape[0])
				if save_image == None:
					save_image = sv_im
				else:
					save_image = torch.cat((save_image, sv_im), dim=2)
			if iteration == (max_firstNBatches_ImageGroups_VisualizationSample - 1):   # when reaching max_step number of iterations/batches in each epoch, the image grid will be saved on the device. The number of image groups in the image grid = firstNBatches * batch_size.
				torchvision.utils.save_image(
					save_image,
					os.path.join(sample_dir, '{}_{}_{}_{}_VisualiationImageGroupsSamples.jpg'.format(config.model_name, config.dataset_name, 'Epoch', epoch))
				)
				
			val_bar.set_description_str('[LLIE] Average PSNR: %.4f dB; Average SSIM: %.4f; Average MAE:  %.4f; Average LPIPS: %.4f' % (
						IQA_metrics_data['average_psnr'], IQA_metrics_data['average_ssim'], IQA_metrics_data['average_mae'], IQA_metrics_data['average_lpips']))
			
		# **Subpart 10: Record the calculated (IQA) metrics to that csv file**
		if (epoch == 0): # if it reaches the first epoch
			with open(csv_result_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
				writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
				writer.writeheader() # The writer writes the header on the csv file
						
		with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
			writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
			writer.writerow(IQA_metrics_data) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file

		if (epoch == config.num_epochs - 1): # if it reaches the last epoch 
			# **Subpart 12: Perform computation complexity metric calculations**
			ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data)

			print('\nAccumulate validation batch duration, for complete training (s):', ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteTraining'])
			print('Accumulate number of enhanced/processed validation input samples, for complete training:', (IQA_metrics_data['accumulate_number_of_val_input_samples_processed'] * config.num_epochs))
			print('Average runtime for enhancing/processing each validation input sample (s):', ComputationalComplexity_metrics_data['average_val_runtime'])

			# **Subpart 13: Record the calculated computation complexity metrics to that csv file**
			with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of ComputationalComplexity_metrics_data dictionary to that csv file.
				writer = csv.DictWriter(csvfile, fieldnames=ComputationalComplexity_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
				writer.writeheader() # The writer writes the header on the csv file
				writer.writerow(ComputationalComplexity_metrics_data)  # The writer writes the data (value) of ComputationalComplexity_metrics_data dictionary in sequence as a row on the csv file

			
		DCE_net.train()
        		


	# -------------------------------------------------------------------
    # train finish
	print("Training completed")

# Task: Make a copy of this script, then->
# 1) At each epoch, try use reduction = sum to directly add psnr_subtotal of each batch, then at last batch only divide the total psnr_subtotal of all batches to get the average psnr of that epoch. Then verify the results by checking the psnr of each image, by using reduction = 'none'. More info: https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/utilities/distributed.py#L22
# 2) Add csv to record the average total loss at each epoch, then plot its graph over epoch and save on device. Same goes to IQA metrics.





