import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import time
import dataloader_WithPair_latest_EMP
import model_ZeroDCE_ori_EMP
import Myloss_ZeroDCE_ori_EMP
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE
from tqdm import tqdm
import torch.utils.data

from calflops import calculate_flops # for calflops to calculate MACs, FLOPs, and trainable parameters
import csv
import math

import pandas as pd
import numpy as np
import random



def load_data(config):
	print('11) For train set:')
	train_dataset = dataloader_WithPair_latest_EMP.LLIEDataset(config.train_GroundTruth_root, config.train_Input_root, config.train_image_size_width, config.train_image_size_height)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
												num_workers=config.num_workers, pin_memory=True)
	print('12) For validation set:')
	val_dataset = dataloader_WithPair_latest_EMP.LLIEDataset(config.val_GroundTruth_root, config.val_Input_root, config.val_image_size_width, config.val_image_size_height)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
												num_workers=config.num_workers, pin_memory=True)

	return train_loader, len(train_loader), val_loader, len(val_loader)

# function used to initialize the parameters (weights and biases) of the network. This function will be used only when the network is not using the pretrained parameters
def weights_init(m): # A custom function that checks the layer type and applies an appropriate initialization strategy for each case, so that it ensures that the initialization is applied consistently across all layers. It iterating through model layers and systematically apply weight initialization across all layers in a model using the model.apply method. Means in this case, all layers whose name containing "Conv" and "BatchNorm" as part will be initialized using the defined methods.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # if layer whose part of its name having 'Conv' is not found by find(), it will return '-1'. So when classname.find('Conv') != -1, it means at least a convolutional layer is found in the model structure, then the parameters of every convolutional layer are set with the predefined ones. The find() method returns the index of first occurrence of the substring (if found). If not found, it returns -1. More info: https://www.programiz.com/python-programming/methods/string/find
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: # if layer whose part of its name having 'BatchNorm' is not found by find(), it will return '-1'. So when classname.find('BatchNorm') != -1, it means at least a convolutional layer is found in the model structure, then the parameters of every BatchNorm layer are set with the predefined ones. The find() method returns the index of first occurrence of the substring (if found). If not found, it returns -1. More info: https://www.programiz.com/python-programming/methods/string/find
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def PSNR(preds, target, data_range=torch.tensor(1.0), base =10.0): # self-written function to return the total PSNR of a batch of images
	
	diff = preds - target # calculate the error of each pixel for each output-input image pair in the batch
	# print('\ndiff:', diff)
	# print('diff.size():\n', diff.size())
	dim=(1,2,3) # Given [N, C, H, W], reduce C,H, and W dimensions = condense all C,H, and W dimensions = sum by N = The remaining dimension is N (batch size)
	sum_squared_error = torch.sum(diff * diff, dim=dim, keepdim=True)
	# print('sum_squared_error:\n', sum_squared_error)
	num_obs = torch.tensor(target[0].numel(), device=target.device) # Get the total number of pixels on an image
	# print('num_obs:\n', num_obs)

	psnr_base_e = 2 * torch.log(data_range) - torch.log(sum_squared_error / num_obs)
	psnr_vals_list = psnr_base_e * (10 / torch.log(torch.tensor(base))) # Get a list of PSNR values of each image pair in that batch
	# print('\npsnr_vals_list:\n', psnr_vals_list)
	batch_sum_psnr = torch.sum(psnr_vals_list).item() # Get the total PSNR values of that batch
	# print('\nbatch_sum_psnr:\n', batch_sum_psnr)
	return batch_sum_psnr

	
def IQA_metrics_calculation(Validation_IQA_metrics_data): # We define Validation_IQA_metrics_data as the input argument, so that the results related to Validation_IQA_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	sample_size = val_LL_image.shape[0] # Get the total number of input samples (img_lowlight.shape[0]) that have been processed by the model in the current iteration, can be equivalent to batch size.
	Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] += sample_size # Update the processed input samples in the current iteration. So that at the end, Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] stores the total number of input samples that have been processed by the model.

	# PSNR part
	batch_sum_psnr = PSNR(val_enhanced_image, val_ori_image) # get the total PSNR of a batch of images
	Validation_IQA_metrics_data['epoch_accumulate_psnr'] += batch_sum_psnr
	Validation_IQA_metrics_data['epoch_average_psnr_db'] = Validation_IQA_metrics_data['epoch_accumulate_psnr'] / Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] # Get the average PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (Summation of PSNR of all image pairs the model has gone through)/[Total batches of images the model has gone through, which is same as the total number of image pairs the model has gone through]).
	Validation_IQA_metrics_data['epoch_average_psnr'] = Validation_IQA_metrics_data['epoch_average_psnr_db']/48.13 # PSNR value is normalized with 48.13 using min max normalization. For 8-bit image, min max normalization = (inst_value - ori_min_value)/(ori_max_value - ori_min_value); where ori_max_value = 20 log (255/255) - 10 log [(1/255)^2]= 48.13dB,  ori_min_value = 20 log (255/255) - 10 log [(255/255)^2]= 0dB


	# SSIM part
	batch_average_ssim = ssim(val_enhanced_image, val_ori_image).item()
	Validation_IQA_metrics_data['epoch_accumulate_ssim'] += batch_average_ssim * sample_size
	Validation_IQA_metrics_data['epoch_average_ssim'] = Validation_IQA_metrics_data['epoch_accumulate_ssim'] / Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed']

	# MAE part
	batch_average_mae = mae(val_enhanced_image, val_ori_image).item()
	Validation_IQA_metrics_data['epoch_accumulate_mae'] += batch_average_mae * sample_size
	Validation_IQA_metrics_data['epoch_average_mae'] = Validation_IQA_metrics_data['epoch_accumulate_mae'] / Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed']
	
	# LPIPS part
	batch_average_lpips = lpips(val_enhanced_image, val_ori_image).item()
	Validation_IQA_metrics_data['epoch_accumulate_lpips'] += batch_average_lpips * sample_size
	Validation_IQA_metrics_data['epoch_average_lpips'] = Validation_IQA_metrics_data['epoch_accumulate_lpips'] / Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed']

	# IQA score part
	Validation_IQA_metrics_data['epoch_average_IQAScore'] = Validation_IQA_metrics_data['epoch_average_psnr'] + Validation_IQA_metrics_data['epoch_average_ssim'] - Validation_IQA_metrics_data['epoch_average_mae'] - Validation_IQA_metrics_data['epoch_average_lpips']



def ComputationComplexity_metrics_calculation(Validation_ComputationalComplexity_metrics_data): # We define Validation_ComputationalComplexity_metrics_data as the input argument, so that the results related to Validation_ComputationalComplexity_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	
	# Calculate the average runtime using the total batch duration and total number of input samples (means the average runtime for an input sample)
	Validation_ComputationalComplexity_metrics_data['average_val_runtime_forCompleteOperations'] = Validation_ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteOperations'] / (Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] * config.num_epochs)
	
	# calflops part: To calculate MACs, FLOPs, and trainable parameters
	# model_name = config.model_name # The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
	batch_size = 1 # reference batch size for calflops
	reference_input_shape = (batch_size, val_LL_image.shape[1], val_LL_image.shape[2], val_LL_image.shape[3]) # (reference batch size for calflops, channel numbers of each input sample, height dimension of each input sample, width dimension of each input sample)
	Validation_ComputationalComplexity_metrics_data['FLOPs'], Validation_ComputationalComplexity_metrics_data['MACs'], Validation_ComputationalComplexity_metrics_data['trainable_parameters'] = calculate_flops(model=DCE_net,  # The model created
																																								input_shape=reference_input_shape,
																																								print_results=False,
                    																																			print_detailed=False,
																																								output_as_string=False,
																																								output_precision=4)

	return reference_input_shape

def save_model(epoch, path, net, optimizer, scheduler, key): # this function saves model checkpoints
	# if not os.path.exists(os.path.join(path, net_name)): # if the folder that stores the checkpoints does not exist
	# 	os.makedirs(os.path.join(path, net_name))

	if key == 0:
		# torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') # The format used by the author. Save the parameters (weigths and biases) of the model at the specified snapshot (EG:epoch) as a pth file	
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints for each config.snap_iter interval
					f=os.path.join(path, '{}-{}-{}-{}-{}-checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name, 'Epoch', epoch)))
	elif key == 1:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_psnr
					f=os.path.join(path, BestEpoAvePSNR_checkpoint_filename))

	elif key == 2:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_ssim
					f=os.path.join(path, BestEpoAveSSIM_checkpoint_filename))

	elif key == 3:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_mae
					f=os.path.join(path, BestEpoAveMAE_checkpoint_filename))

	elif key == 4:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_lpips
					f=os.path.join(path, BestEpoAveLPIPS_checkpoint_filename))
		
	elif key == 5:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_lpips
					f=os.path.join(path, BestValEpoAveLoss_checkpoint_filename))
		
	elif key == 6:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_lpips
					f=os.path.join(path, BestValEpoAveIQAScore_checkpoint_filename))
	
	elif key == 7:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, # save the checkpoints that has the best epoch_average_lpips
					f=os.path.join(path, LastEpoch_checkpoint_filename))


def generate_save_History():

	# Extract losses data from the CSV file part:
	df_TrainingLossesResult = pd.read_csv(csv_TrainingLossesResult_filepath) # convert the csv file into a Dataframe
	
	epoch_list = df_TrainingLossesResult['epoch'].values.tolist() # slice a column from the dataframe, then convert it into a list. 
	epoch_list = [int(i) for i in epoch_list] # convert each string integer number in a list into an integer number.

	epoch_training_average_loss_list = df_TrainingLossesResult['epoch_average_loss'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_training_average_loss_list = [float(i) for i in epoch_training_average_loss_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_training_average_loss_TV_list = df_TrainingLossesResult['epoch_average_Loss_TV'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_training_average_loss_TV_list = [float(i) for i in epoch_training_average_loss_TV_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_training_average_loss_spa_list = df_TrainingLossesResult['epoch_average_loss_spa'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_training_average_loss_spa_list = [float(i) for i in epoch_training_average_loss_spa_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_training_average_loss_col_list = df_TrainingLossesResult['epoch_average_loss_col'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_training_average_loss_col_list = [float(i) for i in epoch_training_average_loss_col_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_training_average_loss_exp_list = df_TrainingLossesResult['epoch_average_loss_exp'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_training_average_loss_exp_list = [float(i) for i in epoch_training_average_loss_exp_list] # convert each string floating point number in a list into a floating point number.
	
	# Extract validation losses results from the CSV file part:
	df_ValidationLossesResult = pd.read_csv(csv_ValidationLossesResult_filepath) # convert the csv file into a Dataframe

	epoch_validation_list = df_ValidationLossesResult['epoch'].values.tolist() # slice a column from the dataframe, then convert it into a list. 
	epoch_validation_list = [int(i) for i in epoch_validation_list] # convert each string integer number in a list into an integer number.

	epoch_validation_average_loss_list = df_ValidationLossesResult['epoch_average_loss'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_loss_list = [float(i) for i in epoch_validation_average_loss_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_validation_average_loss_TV_list = df_ValidationLossesResult['epoch_average_Loss_TV'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_loss_TV_list = [float(i) for i in epoch_validation_average_loss_TV_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_validation_average_loss_spa_list = df_ValidationLossesResult['epoch_average_loss_spa'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_loss_spa_list = [float(i) for i in epoch_validation_average_loss_spa_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_validation_average_loss_col_list = df_ValidationLossesResult['epoch_average_loss_col'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_loss_col_list = [float(i) for i in epoch_validation_average_loss_col_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_validation_average_loss_exp_list = df_ValidationLossesResult['epoch_average_loss_exp'].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_loss_exp_list = [float(i) for i in epoch_validation_average_loss_exp_list] # convert each string floating point number in a list into a floating point number.
	


	# Plot subplots part:
	fig = plt.figure(figsize=(10, 10), dpi=100, constrained_layout=True)
	fig.suptitle('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining and Validation Results: Losses') # set the figure super title
	gs = fig.add_gridspec(3, 2)

	# Generate and save the figure of [Average loss vs Epoch]
	# (Training results) 
	epoch_training_average_loss_list_ymin = min(epoch_training_average_loss_list)
	epoch_training_average_loss_list_xpos = epoch_training_average_loss_list.index(epoch_training_average_loss_list_ymin)
	epoch_training_average_loss_list_xmin = epoch_list[epoch_training_average_loss_list_xpos]
	ax1 = fig.add_subplot(gs[0, :])
	ax1.plot(epoch_list, epoch_training_average_loss_list, 'r--') #row=0, col=0, 1
	ax1.plot(epoch_training_average_loss_list_xmin, epoch_training_average_loss_list_ymin, 'r--', marker='o', fillstyle='none', label='Train') # plot the minimum point
	# (Validation results) 
	epoch_validation_average_loss_list_ymin = min(epoch_validation_average_loss_list)
	epoch_validation_average_loss_list_xpos = epoch_validation_average_loss_list.index(epoch_validation_average_loss_list_ymin)
	epoch_validation_average_loss_list_xmin = epoch_validation_list[epoch_validation_average_loss_list_xpos]
	ax1.plot(epoch_validation_list, epoch_validation_average_loss_list, 'b-') #row=0, col=0, 1
	ax1.plot(epoch_validation_average_loss_list_xmin, epoch_validation_average_loss_list_ymin, 'b-', marker='o', fillstyle='none', label='Validation') # plot the minimum point
	ax1.set_ylabel('Average loss') # set the y-label
	ax1.set_xlabel('Epoch') # set the x-label
	ax1.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax1.set_title(f'Train Y-min. coord.:[{epoch_training_average_loss_list_xmin},{epoch_training_average_loss_list_ymin:.4f}]\n Val Y-min. coord.:[{epoch_validation_average_loss_list_xmin},{epoch_validation_average_loss_list_ymin:.4f}]')
	ax1.legend()
	ax1.grid()

	# Generate and save the figure of [Average loss_TV vs Epoch]
	# (Training results) 
	epoch_training_average_loss_TV_list_ymin = min(epoch_training_average_loss_TV_list)
	epoch_training_average_loss_TV_list_xpos = epoch_training_average_loss_TV_list.index(epoch_training_average_loss_TV_list_ymin)
	epoch_training_average_loss_TV_list_xmin = epoch_list[epoch_training_average_loss_TV_list_xpos]
	ax2 = fig.add_subplot(gs[1, 0])
	ax2.plot(epoch_list, epoch_training_average_loss_TV_list, 'r--') # plot the graph
	ax2.plot(epoch_training_average_loss_TV_list_xmin, epoch_training_average_loss_TV_list_ymin, 'r--', marker='o', fillstyle='none', label='Train') # plot the minimum point
	# (Validation results) 
	epoch_validation_average_loss_TV_list_ymin = min(epoch_validation_average_loss_TV_list)
	epoch_validation_average_loss_TV_list_xpos = epoch_validation_average_loss_TV_list.index(epoch_validation_average_loss_TV_list_ymin)
	epoch_validation_average_loss_TV_list_xmin = epoch_validation_list[epoch_validation_average_loss_TV_list_xpos]
	ax2.plot(epoch_validation_list, epoch_validation_average_loss_TV_list, 'b-') # plot the graph
	ax2.plot(epoch_validation_average_loss_TV_list_xmin, epoch_validation_average_loss_TV_list_ymin, 'b-', marker='o', fillstyle='none', label='Validation') # plot the minimum point
	ax2.set_ylabel('Average loss_TV') # set the y-label
	ax2.set_xlabel('Epoch') # set the x-label
	ax2.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax2.set_title(f'Train Y-min. coord.:[{epoch_training_average_loss_TV_list_xmin},{epoch_training_average_loss_TV_list_ymin:.4f}]\n Val Y-min. coord.:[{epoch_validation_average_loss_TV_list_xmin},{epoch_validation_average_loss_TV_list_ymin:.4f}]')
	ax2.legend()
	ax2.grid()
 	
	# Generate and save the figure of [Average loss_spa vs Epoch]
	# (Training results)
	epoch_training_average_loss_spa_list_ymin = min(epoch_training_average_loss_spa_list)
	epoch_training_average_loss_spa_list_xpos = epoch_training_average_loss_spa_list.index(epoch_training_average_loss_spa_list_ymin)
	epoch_training_average_loss_spa_list_xmin = epoch_list[epoch_training_average_loss_spa_list_xpos]
	ax3 = fig.add_subplot(gs[1, 1])
	ax3.plot(epoch_list, epoch_training_average_loss_spa_list, 'r--') # plot the graph
	ax3.plot(epoch_training_average_loss_spa_list_xmin, epoch_training_average_loss_spa_list_ymin, 'r--', marker='o', fillstyle='none', label='Train') # plot the minimum point
	# (Validation results) 
	epoch_validation_average_loss_spa_list_ymin = min(epoch_validation_average_loss_spa_list)
	epoch_validation_average_loss_spa_list_xpos = epoch_validation_average_loss_spa_list.index(epoch_validation_average_loss_spa_list_ymin)
	epoch_validation_average_loss_spa_list_xmin = epoch_validation_list[epoch_validation_average_loss_spa_list_xpos]
	ax3.plot(epoch_validation_list, epoch_validation_average_loss_spa_list, 'b-') # plot the graph
	ax3.plot(epoch_validation_average_loss_spa_list_xmin, epoch_validation_average_loss_spa_list_ymin, 'b-', marker='o', fillstyle='none', label='Validation') # plot the minimum point
	ax3.set_ylabel('Average loss_spa') # set the y-label
	ax3.set_xlabel('Epoch') # set the x-label
	ax3.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax3.set_title(f'Train Y-min. coord.:[{epoch_training_average_loss_spa_list_xmin},{epoch_training_average_loss_spa_list_ymin:.4f}]\n Val Y-min. coord.:[{epoch_validation_average_loss_spa_list_xmin},{epoch_validation_average_loss_spa_list_ymin:.4f}]')
	ax3.legend()
	ax3.grid()

	# Generate and save the figure of [Average loss_col vs Epoch]
	# (Training results) 
	epoch_training_average_loss_col_list_ymin = min(epoch_training_average_loss_col_list)
	epoch_training_average_loss_col_list_xpos = epoch_training_average_loss_col_list.index(epoch_training_average_loss_col_list_ymin)
	epoch_training_average_loss_col_list_xmin = epoch_list[epoch_training_average_loss_col_list_xpos]
	ax4 = fig.add_subplot(gs[2, 0])
	ax4.plot(epoch_list, epoch_training_average_loss_col_list, 'r--') # plot the graph
	ax4.plot(epoch_training_average_loss_col_list_xmin, epoch_training_average_loss_col_list_ymin, 'r--', marker='o', fillstyle='none', label='Train') # plot the minimum point
	# (Validation results) 
	epoch_validation_average_loss_col_list_ymin = min(epoch_validation_average_loss_col_list)
	epoch_validation_average_loss_col_list_xpos = epoch_validation_average_loss_col_list.index(epoch_validation_average_loss_col_list_ymin)
	epoch_validation_average_loss_col_list_xmin = epoch_validation_list[epoch_validation_average_loss_col_list_xpos]
	ax4.plot(epoch_validation_list, epoch_validation_average_loss_col_list, 'b-') # plot the graph
	ax4.plot(epoch_validation_average_loss_col_list_xmin, epoch_validation_average_loss_col_list_ymin, 'b-', marker='o', fillstyle='none', label='Validation') # plot the minimum point
	ax4.set_ylabel('Average loss_col') # set the y-label
	ax4.set_xlabel('Epoch') # set the x-label
	ax4.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax4.set_title(f'Train Y-min. coord.:[{epoch_training_average_loss_col_list_xmin},{epoch_training_average_loss_col_list_ymin:.4f}]\n Val Y-min. coord.:[{epoch_validation_average_loss_col_list_xmin},{epoch_validation_average_loss_col_list_ymin:.4f}]')
	ax4.legend()
	ax4.grid()

	# Generate and save the figure of [Average loss_exp vs Epoch]
	# (Training results) 
	epoch_training_average_loss_exp_list_ymin = min(epoch_training_average_loss_exp_list)
	epoch_training_average_loss_exp_list_xpos = epoch_training_average_loss_exp_list.index(epoch_training_average_loss_exp_list_ymin)
	epoch_training_average_loss_exp_list_xmin = epoch_list[epoch_training_average_loss_exp_list_xpos]
	ax5 = fig.add_subplot(gs[2, 1])
	ax5.plot(epoch_list, epoch_training_average_loss_exp_list, 'r--') # plot the graph
	ax5.plot(epoch_training_average_loss_exp_list_xmin, epoch_training_average_loss_exp_list_ymin, 'r--', marker='o', fillstyle='none', label='Train') # plot the minimum point
	# (Validation results) 
	epoch_validation_average_loss_exp_list_ymin = min(epoch_validation_average_loss_exp_list)
	epoch_validation_average_loss_exp_list_xpos = epoch_validation_average_loss_exp_list.index(epoch_validation_average_loss_exp_list_ymin)
	epoch_validation_average_loss_exp_list_xmin = epoch_validation_list[epoch_validation_average_loss_exp_list_xpos]
	ax5.plot(epoch_validation_list, epoch_validation_average_loss_exp_list, 'b-') # plot the graph
	ax5.plot(epoch_validation_average_loss_exp_list_xmin, epoch_validation_average_loss_exp_list_ymin, 'b-', marker='o', fillstyle='none', label='Validation') # plot the minimum point
	ax5.set_ylabel('Average loss_exp') # set the y-label
	ax5.set_xlabel('Epoch') # set the x-label
	ax5.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax5.set_title(f'Train Y-min. coord.:[{epoch_training_average_loss_exp_list_xmin},{epoch_training_average_loss_exp_list_ymin:.4f}]\n Val Y-min. coord.:[{epoch_validation_average_loss_exp_list_xmin},{epoch_validation_average_loss_exp_list_ymin:.4f}]')
	ax5.legend()
	ax5.grid()


	epoch_LossesResults_history_filename = '{}-{}-{}-epoch_LossesResults_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	epoch_LossesResults_history_filepath = os.path.join(resultPath_csv, epoch_LossesResults_history_filename) # define the filepath, used to save the figure as an image
	
	plt.margins() 
	plt.savefig(epoch_LossesResults_history_filepath, bbox_inches='tight') # save the figure as an image
	# plt.show()

	# Record the best training losses
	with open(csv_TrainingLossesResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_ComputationalComplexity_metrics_data dictionary to that csv file.
		Losses_fieldnames = ['Fieldnames', 'MinimumEpochAverageLoss', 'MinimumEpochAverageLoss_TV', 'MinimumEpochAverageLoss_Spa', 'MinimumEpochAverageLoss_Col', 'MinimumEpochAverageLoss_Exp']
		best_TrainLosses_HistoryPoint = ['Y-value', epoch_training_average_loss_list_ymin, epoch_training_average_loss_TV_list_ymin, epoch_training_average_loss_spa_list_ymin, epoch_training_average_loss_col_list_ymin, epoch_training_average_loss_exp_list_ymin]
		best_TrainLosses_epoch_location = ['X-value', epoch_training_average_loss_list_xmin, epoch_training_average_loss_TV_list_xmin, epoch_training_average_loss_spa_list_xmin, epoch_training_average_loss_col_list_xmin, epoch_training_average_loss_exp_list_xmin]
		writer = csv.writer(csvfile, delimiter= ',') # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
		writer.writerow(Losses_fieldnames)  
		writer.writerow(best_TrainLosses_HistoryPoint)  
		writer.writerow(best_TrainLosses_epoch_location)  

	# Record the best validation losses
	with open(csv_ValidationLossesResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_ComputationalComplexity_metrics_data dictionary to that csv file.
		Losses_fieldnames = ['Fieldnames', 'MinimumEpochAverageLoss', 'MinimumEpochAverageLoss_TV', 'MinimumEpochAverageLoss_Spa', 'MinimumEpochAverageLoss_Col', 'MinimumEpochAverageLoss_Exp']
		best_ValLosses_HistoryPoint = ['Y-value', epoch_validation_average_loss_list_ymin, epoch_validation_average_loss_TV_list_ymin, epoch_validation_average_loss_spa_list_ymin, epoch_validation_average_loss_col_list_ymin, epoch_validation_average_loss_exp_list_ymin]
		best_ValLosses_epoch_location = ['X-value', epoch_validation_average_loss_list_xmin, epoch_validation_average_loss_TV_list_xmin, epoch_validation_average_loss_spa_list_xmin, epoch_validation_average_loss_col_list_xmin, epoch_validation_average_loss_exp_list_xmin]
		writer = csv.writer(csvfile, delimiter= ',') # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
		writer.writerow(Losses_fieldnames)  
		writer.writerow(best_ValLosses_HistoryPoint)  
		writer.writerow(best_ValLosses_epoch_location) 

	# # (Training results) Generate and save the figure of [Average loss vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_training_average_loss_list:', epoch_training_average_loss_list)
	# x_epoch_training_average_loss_list = [x for x in range(len(epoch_training_average_loss_list))] # create the x-axis elements
	# plt.plot(x_epoch_training_average_loss_list, epoch_training_average_loss_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average loss') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average loss vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_training_average_loss_list), max(x_epoch_training_average_loss_list)+1, 1)) # set the interval of x-axis 
	# epoch_training_average_loss_history_filename = '{}-{}-{}-epoch_training_average_loss_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_training_average_loss_history_filepath = os.path.join(resultPath_csv, epoch_training_average_loss_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_training_average_loss_history_filepath, bbox_inches='tight') # save the figure as an image

	# # (Training results) Generate and save the figure of [Average loss_TV vs Epoch]  **DONE
	# plt.figure() # creates a new figure
	# print('epoch_training_average_loss_TV_list:', epoch_training_average_loss_TV_list)
	# x_epoch_training_average_loss_TV_list = [x for x in range(len(epoch_training_average_loss_TV_list))] # create the x-axis elements
	# plt.plot(x_epoch_training_average_loss_TV_list, epoch_training_average_loss_TV_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average loss_TV') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average loss_TV vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_training_average_loss_TV_list), max(x_epoch_training_average_loss_TV_list)+1, 1)) # set the interval of x-axis 
	# epoch_training_average_loss_TV_history_filename = '{}-{}-{}-epoch_training_average_loss_TV_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_training_average_loss_TV_history_filepath = os.path.join(resultPath_csv, epoch_training_average_loss_TV_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_training_average_loss_TV_history_filepath, bbox_inches='tight') # save the figure as an image

	# # (Training results) Generate and save the figure of [Average loss_spa vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_training_average_loss_spa_list:', epoch_training_average_loss_spa_list)
	# x_epoch_training_average_loss_spa_list = [x for x in range(len(epoch_training_average_loss_spa_list))] # create the x-axis elements
	# plt.plot(x_epoch_training_average_loss_spa_list, epoch_training_average_loss_spa_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average loss_spa') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average loss_spa vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_training_average_loss_spa_list), max(x_epoch_training_average_loss_spa_list)+1, 1)) # set the interval of x-axis 
	# epoch_training_average_loss_spa_history_filename = '{}-{}-{}-epoch_training_average_loss_spa_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_training_average_loss_spa_history_filepath = os.path.join(resultPath_csv, epoch_training_average_loss_spa_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_training_average_loss_spa_history_filepath, bbox_inches='tight') # save the figure as an image
	
	# # (Training results) Generate and save the figure of [Average loss_col vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_training_average_loss_col_list:', epoch_training_average_loss_col_list)
	# x_epoch_training_average_loss_col_list = [x for x in range(len(epoch_training_average_loss_col_list))] # create the x-axis elements
	# plt.plot(x_epoch_training_average_loss_col_list, epoch_training_average_loss_col_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average loss_col') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average loss_col vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_training_average_loss_col_list), max(x_epoch_training_average_loss_col_list)+1, 1)) # set the interval of x-axis 
	# epoch_training_average_loss_col_history_filename = '{}-{}-{}-epoch_training_average_loss_col_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_training_average_loss_col_history_filepath = os.path.join(resultPath_csv, epoch_training_average_loss_col_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_training_average_loss_col_history_filepath, bbox_inches='tight') # save the figure as an image

	# # (Training results) Generate and save the figure of [Average loss_exp vs Epoch]
	# plt.figure() # creates a new figure
	# print('epoch_training_average_loss_exp_list:', epoch_training_average_loss_exp_list)
	# x_epoch_training_average_loss_exp_list = [x for x in range(len(epoch_training_average_loss_exp_list))] # create the x-axis elements
	# plt.plot(x_epoch_training_average_loss_exp_list, epoch_training_average_loss_exp_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average loss_exp') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average loss_exp vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_training_average_loss_exp_list), max(x_epoch_training_average_loss_exp_list)+1, 1)) # set the interval of x-axis 
	# epoch_training_average_loss_exp_history_filename = '{}-{}-{}-epoch_training_average_loss_exp_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_training_average_loss_exp_history_filepath = os.path.join(resultPath_csv, epoch_training_average_loss_exp_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_training_average_loss_exp_history_filepath, bbox_inches='tight') # save the figure as an image

	# Extract validation IQA results from the CSV file part:
	df_ValidationResult = pd.read_csv(csv_ValidationIQAResult_filepath) # convert the csv file into a Dataframe
	# print("df_ValidationResult:", df_ValidationResult)
	epoch_list = df_ValidationResult['epoch'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_list = [int(i) for i in epoch_list] # convert each string integer number in a list into an integer number.

	epoch_validation_average_IQAScore_list = df_ValidationResult['epoch_average_IQAScore'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_IQAScore_list = [float(i) for i in epoch_validation_average_IQAScore_list] # convert each string floating point number in a list into a floating point number.

	epoch_validation_average_psnr_db_list = df_ValidationResult['epoch_average_psnr_db'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_psnr_db_list = [float(i) for i in epoch_validation_average_psnr_db_list] # convert each string floating point number in a list into a floating point number.

	epoch_validation_average_psnr_list = df_ValidationResult['epoch_average_psnr'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_psnr_list = [float(i) for i in epoch_validation_average_psnr_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_validation_average_ssim_list = df_ValidationResult['epoch_average_ssim'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_ssim_list = [float(i) for i in epoch_validation_average_ssim_list] # convert each string floating point number in a list into a floating point number.
	
	epoch_validation_average_mae_list = df_ValidationResult['epoch_average_mae'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_mae_list = [float(i) for i in epoch_validation_average_mae_list] # convert each string floating point number in a list into a floating point number.

	epoch_validation_average_lpips_list = df_ValidationResult['epoch_average_lpips'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
	epoch_validation_average_lpips_list = [float(i) for i in epoch_validation_average_lpips_list] # convert each string floating point number in a list into a floating point number.

	# Plot subplots part:
	fig = plt.figure(figsize=(10, 10), dpi=100, constrained_layout=True)
	fig.suptitle('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nValidation Results: IQA Metrics') # set the figure super title
	gs = fig.add_gridspec(3, 2) # create 3x2 grid

	# (Validation results) Generate and save the figure of [Average IQA Score vs Epoch]
	epoch_validation_average_IQAScore_list_ymax = max(epoch_validation_average_IQAScore_list)
	epoch_validation_average_IQAScore_list_xpos = epoch_validation_average_IQAScore_list.index(epoch_validation_average_IQAScore_list_ymax)
	epoch_validation_average_IQAScore_list_xmax = epoch_list[epoch_validation_average_IQAScore_list_xpos]
	ax6 = fig.add_subplot(gs[0, :])
	ax6.plot(epoch_list, epoch_training_average_loss_list, 'g--') #row=0, col=0, 1
	ax6.plot(epoch_training_average_loss_list_xmin, epoch_training_average_loss_list_ymin, 'g--', marker='o', fillstyle='none', label='Average loss [Train]') # plot the minimum point
	ax6.plot(epoch_validation_list, epoch_validation_average_loss_list, 'r-') #row=0, col=0, 1
	ax6.plot(epoch_validation_average_loss_list_xmin, epoch_validation_average_loss_list_ymin, 'r-', marker='o', fillstyle='none', label='Average loss [Val]') # plot the minimum point
	ax6.plot(epoch_list, epoch_validation_average_IQAScore_list, 'b--')
	ax6.plot(epoch_validation_average_IQAScore_list_xmax, epoch_validation_average_IQAScore_list_ymax, 'b--', marker='o', fillstyle='none', label = 'Average IQA score') # plot the maximum point
	ax6.set_ylabel('Average IQA Score or Average loss') # set the y-label
	ax6.set_xlabel('Epoch') # set the x-label
	ax6.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax6.set_title(f'Average IQA score: Y-max. coord.:[{epoch_validation_average_IQAScore_list_xmax},{epoch_validation_average_IQAScore_list_ymax:.4f}]\nAverage loss [Train]: Y-min. coord.:[{epoch_training_average_loss_list_xmin},{epoch_training_average_loss_list_ymin:.4f}]\nAverage loss [Val]: Y-min. coord.:[{epoch_validation_average_loss_list_xmin},{epoch_validation_average_loss_list_ymin:.4f}]')
	ax6.legend()
	ax6.grid()

	# (Validation results) Generate and save the figure of [Average PSNR (dB) vs Epoch]
	epoch_validation_average_psnr_db_list_ymax = max(epoch_validation_average_psnr_db_list)
	epoch_validation_average_psnr_db_list_xpos = epoch_validation_average_psnr_db_list.index(epoch_validation_average_psnr_db_list_ymax)
	epoch_validation_average_psnr_db_list_xmax = epoch_list[epoch_validation_average_psnr_db_list_xpos]
	ax7 = fig.add_subplot(gs[1, 0])
	ax7.plot(epoch_list, epoch_validation_average_psnr_db_list, 'b') #row=0, col=0, 1
	ax7.plot(epoch_validation_average_psnr_db_list_xmax, epoch_validation_average_psnr_db_list_ymax, 'b', marker='o', fillstyle='none') # plot the maximum point
	ax7.set_ylabel('Average PSNR [dB]') # set the y-label
	ax7.set_xlabel('Epoch') # set the x-label
	ax7.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax7.set_title(f'Y-max. coord.:[{epoch_validation_average_psnr_db_list_xmax},{epoch_validation_average_psnr_db_list_ymax:.4f}]')
	ax7.grid()
	

	# (Validation results) Generate and save the figure of [Average SSIM vs Epoch]
	epoch_validation_average_ssim_list_ymax = max(epoch_validation_average_ssim_list)
	epoch_validation_average_ssim_list_xpos = epoch_validation_average_ssim_list.index(epoch_validation_average_ssim_list_ymax)
	epoch_validation_average_ssim_list_xmax = epoch_list[epoch_validation_average_ssim_list_xpos]
	ax8 = fig.add_subplot(gs[1, 1])
	ax8.plot(epoch_list, epoch_validation_average_ssim_list, 'b') #row=0, col=0, 1
	ax8.plot(epoch_validation_average_ssim_list_xmax, epoch_validation_average_ssim_list_ymax, 'b', marker='o', fillstyle='none') # plot the maximum point
	ax8.set_ylabel('Average SSIM') # set the y-label
	ax8.set_xlabel('Epoch') # set the x-label
	ax8.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax8.set_title(f'Y-max. coord.:[{epoch_validation_average_ssim_list_xmax},{epoch_validation_average_ssim_list_ymax:.4f}]')
	ax8.grid()

	# (Validation results) Generate and save the figure of [Average MAE vs Epoch]
	epoch_validation_average_mae_list_ymin = min(epoch_validation_average_mae_list)
	epoch_validation_average_mae_list_xpos = epoch_validation_average_mae_list.index(epoch_validation_average_mae_list_ymin)
	epoch_validation_average_mae_list_xmin = epoch_list[epoch_validation_average_mae_list_xpos]
	ax9 = fig.add_subplot(gs[2, 0])
	ax9.plot(epoch_list, epoch_validation_average_mae_list, 'b') #row=0, col=0, 1
	ax9.plot(epoch_validation_average_mae_list_xmin, epoch_validation_average_mae_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax9.set_ylabel('Average MAE') # set the y-label
	ax9.set_xlabel('Epoch') # set the x-label
	ax9.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax9.set_title(f'Y-min. coord.:[{epoch_validation_average_mae_list_xmin},{epoch_validation_average_mae_list_ymin:.4f}]')
	ax9.grid()

	# (Validation results) Generate and save the figure of [Average LPIPS vs Epoch]
	epoch_validation_average_lpips_list_ymin = min(epoch_validation_average_lpips_list)
	epoch_validation_average_lpips_list_xpos = epoch_validation_average_lpips_list.index(epoch_validation_average_lpips_list_ymin)
	epoch_validation_average_lpips_list_xmin = epoch_list[epoch_validation_average_lpips_list_xpos]
	ax10 = fig.add_subplot(gs[2, 1])
	ax10.plot(epoch_list, epoch_validation_average_lpips_list, 'b') #row=0, col=0, 1
	ax10.plot(epoch_validation_average_lpips_list_xmin, epoch_validation_average_lpips_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax10.set_ylabel('Average LPIPS') # set the y-label
	ax10.set_xlabel('Epoch') # set the x-label
	ax10.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, math.ceil(config.num_epochs*0.2))) # set the interval of x-axis 
	ax10.set_title(f'Y-min. coord.:[{epoch_validation_average_lpips_list_xmin},{epoch_validation_average_lpips_list_ymin:.4f}]')
	ax10.grid()

	epoch_validation_results_filename = '{}-{}-{}-epoch_validation_results_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	epoch_validation_results_filepath = os.path.join(resultPath_csv, epoch_validation_results_filename) # define the filepath, used to save the figure as an image

	plt.margins() 
	plt.savefig(epoch_validation_results_filepath, bbox_inches='tight') # save the figure as an image
	# plt.show()

	with open(csv_ValidationIQAResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_ComputationalComplexity_metrics_data dictionary to that csv file.
		IQA_fieldnames = ['Fieldnames','MaximumEpochAverageIQAScore', 'MaximumEpochAveragePSNR_dB', 'MaximumEpochAverageSSIM', 'MinimumEpochAverageMAE', 'MinimumEpochAverageLPIPS']
		best_IQAMetrics_HistoryPoint = ['Y-value', epoch_validation_average_IQAScore_list_ymax, epoch_validation_average_psnr_db_list_ymax, epoch_validation_average_ssim_list_ymax, epoch_validation_average_mae_list_ymin, epoch_validation_average_lpips_list_ymin]
		best_IQAMetrics_epoch_location = ['X-value', epoch_validation_average_IQAScore_list_xmax, epoch_validation_average_psnr_db_list_xmax, epoch_validation_average_ssim_list_xmax, epoch_validation_average_mae_list_xmin, epoch_validation_average_lpips_list_xmin]
		writer = csv.writer(csvfile, delimiter= ',') # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
		writer.writerow(IQA_fieldnames)  
		writer.writerow(best_IQAMetrics_HistoryPoint)  
		writer.writerow(best_IQAMetrics_epoch_location)  


	# # (Validation results) Generate and save the figure of [Average PSNR vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_validation_average_psnr_list:', epoch_validation_average_psnr_list)
	# x_epoch_validation_average_psnr_list = [x for x in range(len(epoch_validation_average_psnr_list))] # create the x-axis elements
	# plt.plot(x_epoch_validation_average_psnr_list, epoch_validation_average_psnr_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average PSNR [dB]') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average PSNR vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_validation_average_psnr_list), max(x_epoch_validation_average_psnr_list)+1, 1)) # set the interval of x-axis 
	# epoch_validation_average_psnr_history_filename = '{}-{}-{}-epoch_validation_average_psnr_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_validation_average_psnr_history_filepath = os.path.join(resultPath_csv, epoch_validation_average_psnr_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_validation_average_psnr_history_filepath, bbox_inches='tight') # save the figure as an image

	# # (Validation results) Generate and save the figure of [Average SSIM vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_validation_average_ssim_list:', epoch_validation_average_ssim_list)
	# x_epoch_validation_average_ssim_list = [x for x in range(len(epoch_validation_average_ssim_list))] # create the x-axis elements
	# plt.plot(x_epoch_validation_average_ssim_list, epoch_validation_average_ssim_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average SSIM') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average SSIM vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_validation_average_ssim_list), max(x_epoch_validation_average_ssim_list)+1, 1)) # set the interval of x-axis 
	# epoch_validation_average_ssim_history_filename = '{}-{}-{}-epoch_validation_average_ssim_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_validation_average_ssim_history_filepath = os.path.join(resultPath_csv, epoch_validation_average_ssim_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_validation_average_ssim_history_filepath, bbox_inches='tight') # save the figure as an image
	
	# # (Validation results) Generate and save the figure of [Average MAE vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_validation_average_mae_list:', epoch_validation_average_mae_list)
	# x_epoch_validation_average_mae_list = [x for x in range(len(epoch_validation_average_mae_list))] # create the x-axis elements
	# plt.plot(x_epoch_validation_average_mae_list, epoch_validation_average_mae_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average MAE') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average MAE vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_validation_average_mae_list), max(x_epoch_validation_average_mae_list)+1, 1)) # set the interval of x-axis 
	# epoch_validation_average_mae_history_filename = '{}-{}-{}-epoch_validation_average_mae_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_validation_average_mae_history_filepath = os.path.join(resultPath_csv, epoch_validation_average_mae_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_validation_average_mae_history_filepath, bbox_inches='tight') # save the figure as an image

	# # (Validation results) Generate and save the figure of [Average LPIPS vs Epoch] **DONE
	# plt.figure() # creates a new figure
	# print('epoch_validation_average_lpips_list:', epoch_validation_average_lpips_list)
	# x_epoch_validation_average_lpips_list = [x for x in range(len(epoch_validation_average_lpips_list))] # create the x-axis elements
	# plt.plot(x_epoch_validation_average_lpips_list, epoch_validation_average_lpips_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average LPIPS') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average LPIPS vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_validation_average_lpips_list), max(x_epoch_validation_average_lpips_list)+1, 1)) # set the interval of x-axis 
	# epoch_validation_average_lpips_history_filename = '{}-{}-{}-epoch_validation_average_lpips_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_validation_average_lpips_history_filepath = os.path.join(resultPath_csv, epoch_validation_average_lpips_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_validation_average_lpips_history_filepath, bbox_inches='tight') # save the figure as an image

	
	





# **Part 1: ArgumentParser, usable only if you run this script**
if __name__ == '__main__':

	parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
	
	# Train & Validation configurations
	# Directories and names for Train & Validation 
	parser.add_argument('--model_name', type=str, default= "ZeroDCE_ori") # Max 30 characters. The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
	parser.add_argument('--dataset_name', type=str, default= "SICE_Part1") # Max 15 characters
	parser.add_argument('--train_GroundTruth_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/train/GroundTruth") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--train_Input_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/train/Input") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--val_GroundTruth_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/val/GroundTruth") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--val_Input_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/val/Input") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--dir_store_results', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/ZeroDCE_ori_EMP") 
	# parser.add_argument('--pretrain_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/snapshots/Epoch99.pth") # The pretrained model parameters provided by the author
	parser.add_argument('--checkpoint_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/results/model_parameters/Zero-DCE/enhance_1.pt") # The pretrained model parameters obtained by myself
	parser.add_argument('--snapshot_iter', type=int, default=1) # The interval to save the checkpoint
	parser.add_argument('--BatchesNum_ImageGroups_VisualizationSample', type=int, default=1) # The number of image groups in the image grid = firstNBatches * batch_size during validation.: At each epoch, save the maximum first N batches of val_LL_image (Input), val_enhanced_image (Output), and val_ori_image (GroundTruth) image groups in validation as the samples to visualize the network performance. Max can have 32 images (image grid = 1 * 32) of 256x256 dimensions concatenated on a horizontal line for each epoch. 
	
	# Train & Validation hyperparameters
	parser.add_argument('--load_pretrain', type=bool, default= False) # Set if want to load the pretrained model's parameter. "False" means does not load the pretrained model's parameters; "True" means load the pretrained model's parameters
	parser.add_argument('--train_image_size_height', type=int, default=256) # The height size of the input train images to be resized (in pixel dimension)
	parser.add_argument('--train_image_size_width', type=int, default=256) # The width size of the input train images to be resized (in pixel dimension)
	parser.add_argument('--val_image_size_height', type=int, default=256) # The height size of the input validation images to be resized (in pixel dimension)
	parser.add_argument('--val_image_size_width', type=int, default=256) # The width size of the input validation images to be resized (in pixel dimension)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=0.0001) # Learning rate. Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--scheduler_step_size', type=int, default = 125) # The step size of scheduler: Period of learning rate decay. EG: value = num_epochs*0.3 to use 3 different learning rates along the training; value = num_epochs*0.5 to use 2 different learning rates along the training...
	parser.add_argument('--scheduler_gamma', type=float, default = 1) # Multiplicative factor of learning rate decay (EG: _last_lr = lr*scheduler_gamma). "1" means disable scheduler; else, means enable scheduler with that gamma
	parser.add_argument('--earlystopping_switch', type=bool, default= False) # Set if want to enable early stopping. "False" means disable early stopping; "True" means enable early stopping  
	parser.add_argument('--earlystopping_patience', type=int, default = 100) # the threshold (the number of consecutive epoch for no improvement on all IQA metrics) to wait before stopping model training, when there are consecutives no improvements on all IQA metrics
	
	# Model parameters
	parser.add_argument('--weightage_loss_TV', type=float, default = 20) # the weightage of loss_TV
	parser.add_argument('--weightage_loss_spa', type=float, default = 1) # the weightage of loss_spa
	parser.add_argument('--weightage_loss_col', type=float, default = 0.5) # the weightage of loss_col
	parser.add_argument('--weightage_loss_exp', type=float, default = 1) # the weightage of loss_exp
	

	# The parse_args() object (config=self) takes the data (values) you provide to your positional/optional arguments on command line interface or within the () of parse_args(), then converts them into the required data type as mentioned in add_argument() respectively. 
	# So you can access the data of a positional/optional argument by using the syntax args.argument_name (EG: config.lowlight_images_path).
	config = parser.parse_args() 

### **Part 2: Initialization**


# **Subpart 1: Select the device for computations**
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	random.seed(1) # For deterministic/reproducible results:
	np.random.seed(1) # For deterministic/reproducible results:
	torch.manual_seed(1) # For deterministic/reproducible results: This function ensures the reproducibility of the parameters (weights and biases) of the network which are initialized using random normal functions. The PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch)
	torch.cuda.manual_seed_all(1) # For deterministic/reproducible results:
	torch.backends.cudnn.deterministic=True # For deterministic/reproducible results: Ensures that cuDNN uses deterministic algorithms for convolution operations
	torch.backends.cudnn.benchmark=False # For deterministic/reproducible results: Ensures that CUDA selects the same algorithm each time an application is run

	print('-------LLIE model training and validation configurations-------')
	print('1) Model name:', config.model_name)
	print('2) Dataset name:', config.dataset_name)
	print('3) Using pretrained model?:', config.load_pretrain)
	print('4) Directory containing train set [GroundTruth]:', config.train_GroundTruth_root)
	print('5) Directory containing train set [Input]:', config.train_Input_root)
	print('6) Directory containing validation set [GroundTruth]:', config.val_GroundTruth_root)
	print('7) Directory containing validation set [Input]:', config.val_Input_root)
	print('8) Directory to store training results:', config.dir_store_results)
	print('9) The images in train set will be resized to (height x width, pixel dimensions): {} x {}'.format(config.train_image_size_height, config.train_image_size_width))
	print('10) The images in validation set will be resized to (height x width, pixel dimensions): {} x {}'.format(config.val_image_size_height, config.val_image_size_width))
	

# **Initialize folders
	# **Subpart 5: Define the paths/folders that store the results obtained from this script**
	current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
	resultPath = config.dir_store_results + '/' + '{}-{}-TrainResultsPool'.format(current_date_time_string, config.dataset_name) # Create a new path by changing a part of resultPath
	os.makedirs(resultPath, exist_ok=True)
	# resultPath_subfolder = "self_CompileMetrics" # Define the folder that stores the enhanced images and the grids of input-output image pairs involved in metric calculations as different folder respectively
		
	# # For the folder that stores the enhanced images
	# resultPath_subfolder_individual = "EnhancedImages" # Define the folder that stores the enhanced images
	# resultPath_EnhancedResults = os.path.join(resultPath, resultPath_subfolder_individual) # Define the absolute path to the folder that stores the enhanced images
	# os.makedirs(resultPath_EnhancedResults, exist_ok=True) # Ensure the path that stores the enhanced images is created
		
	# # For the folder that stores the grids of input-output image pairs involved in metric calculations
	# resultPath_subfolder_imagepairs = "ImagePairs" # Define the folder that stores the grids of input-output image pairs involved in metric calculations
	# resultPath_ImagePairsResults = os.path.join(resultPath, resultPath_subfolder_imagepairs) # Define the absolute path to the folder that stores the grids of input-output image pairs involved in metric calculations
	# os.makedirs(resultPath_ImagePairsResults, exist_ok=True) # Ensure the path that stores the grids of input-output image pairs involved in metric calculations is created
	

	# For the folder that stores the checkpoints containing the states of epoch, model parameters, optimizer, and loss [Used to apply on model for inference/resume training]
	resultPath_subfolder_modelparameters = "ModelParam" # Define the folder that stores the grids of input-output image pairs involved in metric calculations
	resultPath_ModelParametersResults = os.path.join(resultPath, resultPath_subfolder_modelparameters) # Define the absolute path to the folder that stores the grids of input-output image pairs involved in metric calculations
	os.makedirs(resultPath_ModelParametersResults, exist_ok=True) # Ensure the path that stores the grids of input-output image pairs involved in metric calculations is created
	
	# For the folder that stores the training and validation results history
	# For the csv that stores the metrics data
	resultPath_subfolder_csv = "csvFile" # Define the folder that stores the csv file
	resultPath_csv = os.path.join(resultPath, resultPath_subfolder_csv) # Define the absolute path to the folder that stores the csv file
	os.makedirs(resultPath_csv, exist_ok=True) # Ensure the path that stores the csv file is created
	

	# To record training results 
	csv_TrainingLossesResult_filename = "{}-{}-{}-TrainLossesHistory.csv".format(current_date_time_string, config.model_name, config.dataset_name) # Create the filename of the csv that stores the metrics data
	csv_TrainingLossesResult_filepath = os.path.join(resultPath_csv, csv_TrainingLossesResult_filename) # Create the path to the csv that stores the metrics data

	# To record validation results 
	csv_ValidationIQAResult_filename = "{}-{}-{}-ValIQAHistory.csv".format(current_date_time_string, config.model_name, config.dataset_name) # Create the filename of the csv that stores the metrics data
	csv_ValidationIQAResult_filepath = os.path.join(resultPath_csv, csv_ValidationIQAResult_filename) # Create the path to the csv that stores the metrics data
	# with open(csv_ValidationResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
	# 	writer = csv.DictWriter(csvfile, fieldnames=Validation_IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_IQA_metrics_data.keys() as the elements=keys of the header
	# 	writer.writeheader() # The writer writes the header on the csv file

	csv_ValidationLossesResult_filename = "{}-{}-{}-ValLossesHistory.csv".format(current_date_time_string, config.model_name, config.dataset_name) # Create the filename of the csv that stores the metrics data
	csv_ValidationLossesResult_filepath = os.path.join(resultPath_csv, csv_ValidationLossesResult_filename) # Create the path to the csv that stores the metrics data

	csv_configuration_filename = "{}-{}-{}-TrainValConfiguration.csv".format(current_date_time_string, config.model_name, config.dataset_name) # Create the filename of the csv that stores the metrics data
	csv_configuration_filepath = os.path.join(resultPath_csv, csv_configuration_filename) # Create the path to the csv that stores the metrics data

	# For the folder that stores the grids of input-output-GroundTruth image groups involved in metric calculations
	resultPath_subfolder_sample_val_output_folder = "SampleValOutput" # Define the folder that stores the validation output samples
	sample_output_folder = os.path.join(resultPath, resultPath_subfolder_sample_val_output_folder) # Create a new path by changing a part of resultPath
	os.makedirs(sample_output_folder, exist_ok=True) # Ensure the path that stores the validation output samples is created
	
	# Initialize filenames to save different best checkpoints
	BestEpoAvePSNR_checkpoint_filename = '{}-{}-{}-BestEpoAvePSNR_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)
	BestEpoAveSSIM_checkpoint_filename = '{}-{}-{}-BestEpoAveSSIM_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)
	BestEpoAveMAE_checkpoint_filename = '{}-{}-{}-BestEpoAveMAE_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)
	BestEpoAveLPIPS_checkpoint_filename = '{}-{}-{}-BestEpoAveLPIPS_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)
	BestValEpoAveLoss_checkpoint_filename = '{}-{}-{}-BestValEpoAveLoss_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)
	BestValEpoAveIQAScore_checkpoint_filename = '{}-{}-{}-BestValEpoAveIQAScore_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)
	LastEpoch_checkpoint_filename = '{}-{}-{}-LastEpoch_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)


	# Record training configurations
	with open(csv_configuration_filepath, 'w', newline='') as csvfile: 
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['-------LLIE model training and validation configurations-------'])
		config_dict = vars(config) # vars() returns the __dict__ attribute for a module, class, instance, or any other object if the same has a __dict__ attribute. Since config has __dict__ attribute, vars(config) returns all the key-value pairs of config as a dictionary 
		for key, value in config_dict.items():
			writer.writerow([key, value]) # The writer writes the header on the csv file
		
		

# **Subpart 2: Initialize training and validation dataset**
	train_loader, train_number, val_loader, val_number = load_data(config)


# **Subpart 3: Initialize the network**
	DCE_net = model_ZeroDCE_ori_EMP.enhance_net_nopool().cuda() # Move the model to the specified device so that the samples in each batch can be processed simultaneously

	DCE_net.apply(weights_init)
	if config.load_pretrain == True: # When using the pretrained network parameters
		# DCE_net.load_state_dict(torch.load(config.pretrain_dir)) # load the pretrained parameters (weights & biases) of the model obtained/learned at the snapshot (EG: at a particular epoch)
		checkpoint = torch.load(config.checkpoint_dir)
		DCE_net.load_state_dict(checkpoint['model_state_dict']) # load the parameters (weights & biases) of the model obtained/learned at a particular checkpoint (EG: at a particular epoch)


# **Subpart 4: Initialize the loss functions used for training**
	L_color = Myloss_ZeroDCE_ori_EMP.L_color()
	L_spa = Myloss_ZeroDCE_ori_EMP.L_spa()
	L_exp = Myloss_ZeroDCE_ori_EMP.L_exp(16,0.6)
	L_TV = Myloss_ZeroDCE_ori_EMP.L_TV()

# **Subpart 5: Initialize the optimizer used for training**
	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay) # Set the hyperparameters for the optimizer to adjust the model parameters (weights and biases), after each loss.backward() [perform backpropagation = calculate the partial derivative of loss with respect to each weight] of a batch of samples
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma, last_epoch=-1) # Decays the learning rate of each parameter group by gamma every step_size epochs.
 
# **Subpart 6: Initialize the IQA metric-related functions to calculate IQA results during validation**
	ssim = SSIM(data_range=1.).cuda() # Performance metric: SSIM
	mae = MAE(num_outputs=1).cuda() # Performance metric: MAE
	lpips = LPIPS(net_type='alex', normalize=True).cuda() # Performance metric: LPIPS (perceptual loss provided by a pretrained learning-based model), using alexnet as the backbone (LPIPS with alexnet performs the best according to its official Github)

# ** Initialize the variables and threshold related to early stopping
	patience = config.earlystopping_patience # the threshold (the number of consecutive epoch for no improvement on all IQA metrics) to wait before stopping model training, when there are consecutives no improvements on all IQA metrics
	wait = 0
	

### **Part 3: Training**
	print('\n-------Operations begin-------')
	DCE_net.train()

	# **Subpart 2: Initialize a dictionary to store the computational complexity metric data , so they will be updated from time to time later**
	Validation_ComputationalComplexity_metrics_data ={'average_val_runtime_forCompleteOperations':0., 'trainable_parameters':0., 'MACs':0., 'FLOPs': 0., 'accumulate_val_batch_duration_forCompleteOperations':0.}
	
	stopped_epoch = -1 # if stopped_epoch printed as -1 means early stopping does not occur; else, the value represents the epoch number the early stopping occurs

	for epoch in range(config.num_epochs):

		Training_losses_data ={'epoch':0, 'epoch_average_loss':0., 'epoch_average_Loss_TV':0., 'epoch_average_loss_spa':0., 'epoch_average_loss_col': 0., 'epoch_average_loss_exp':0., 'epoch_accumulate_number_of_training_input_samples_processed': 0, 'epoch_accumulate_loss':0., 'epoch_accumulate_Loss_TV':0., 'epoch_accumulate_loss_spa':0., 'epoch_accumulate_loss_col': 0., 'epoch_accumulate_loss_exp':0.}
		Training_losses_data['epoch'] = epoch # Update the current epoch (The number of epoch and batch, when necessary) to the Validation_IQA_metrics_data dictionary

		# **Subpart 1: Initialize a dictionary to store the Image Quality Assessment (IQA) metric data , so they will be updated from time to time later**
		Validation_IQA_metrics_data = {'epoch':0, 'epoch_average_IQAScore': 0., 'epoch_average_psnr_db': 0., 'epoch_average_psnr': 0.,'epoch_average_ssim': 0., 'epoch_average_mae': 0., 'epoch_average_lpips': 0., 'epoch_accumulate_number_of_val_input_samples_processed': 0., 'epoch_accumulate_psnr': 0., 'epoch_accumulate_ssim': 0., 'epoch_accumulate_mae': 0.,'epoch_accumulate_lpips': 0.}
		Validation_IQA_metrics_data['epoch'] = epoch # Update the current epoch (The number of epoch and batch, when necessary) to the Validation_IQA_metrics_data dictionary

		Validation_losses_data ={'epoch':0, 'epoch_average_loss':0., 'epoch_average_Loss_TV':0., 'epoch_average_loss_spa':0., 'epoch_average_loss_col': 0., 'epoch_average_loss_exp':0., 'epoch_accumulate_number_of_training_input_samples_processed': 0, 'epoch_accumulate_loss':0., 'epoch_accumulate_Loss_TV':0., 'epoch_accumulate_loss_spa':0., 'epoch_accumulate_loss_col': 0., 'epoch_accumulate_loss_exp':0.}
		Validation_losses_data['epoch'] = epoch # Update the current epoch (The number of epoch and batch, when necessary) to the Validation_IQA_metrics_data dictionary

		print('Epoch: {}/{} | Model training begins'.format(Training_losses_data['epoch'] + 1, config.num_epochs))
		train_bar = tqdm(train_loader) # tqdm automatically detects the length of the iterable object (train_loader) [The length of the iterable object = The total number of batches of samples] and generates a progress bar that updates dynamically as each item=(batch of samples) is processed.
		for iteration, (ori_image, LL_image) in enumerate(train_bar):
			# count = epoch * train_number + (iteration + 1)
			ori_image, LL_image = ori_image.cuda(), LL_image.cuda() # The model requires normallight-lowlight image pairs as inputs          
			# print('Train: ori_image.shape():', ori_image.shape) # show the shape of train images
			# print('ori_image:', ori_image)
			# print('LL_image:', LL_image)
			enhanced_image_1,enhanced_image,A = DCE_net(LL_image) # Call the model to enhance this batch of samples, by returning each image enhanced at 4th enhancing iteration, each image enhanced at last enhancing iteration, and the 24 curve parameter maps used to enhance this batch of samples at different stages respectively.
				
			loss_TV = config.weightage_loss_TV*L_TV(A) # Calculate the illumination smoothness loss of this batch of samples
			loss_spa = config.weightage_loss_spa*torch.mean(L_spa(enhanced_image, LL_image)) # Calculate the spatial consistency loss of this batch of samples
			loss_col = config.weightage_loss_col*torch.mean(L_color(enhanced_image)) # Calculate the color constancy loss of this batch of samples
			loss_exp = config.weightage_loss_exp*torch.mean(L_exp(enhanced_image)) # Calculate the exposure control loss of this batch of samples
			# Total loss of this batch of samples (best_loss, considering the weightage of each loss contributed to the total loss)
			loss =  loss_TV + loss_spa + loss_col + loss_exp 

			# The epoch losses for record purpose only
			Training_losses_data['epoch_accumulate_loss'] = Training_losses_data['epoch_accumulate_loss'] + loss.item() # epoch_accumulate_loss stores the accumulate loss in an epoch, for record purpose only (not for backpropagation)
			Training_losses_data['epoch_average_loss'] = Training_losses_data['epoch_accumulate_loss']/(iteration+1)
			Training_losses_data['epoch_accumulate_Loss_TV'] = Training_losses_data['epoch_accumulate_Loss_TV'] + loss_TV.item() # epoch_accumulate_Loss_TV stores the accumulate Loss_TV in an epoch, for record purpose only (not for backpropagation)
			Training_losses_data['epoch_average_Loss_TV'] = Training_losses_data['epoch_accumulate_Loss_TV']/(iteration+1)
			Training_losses_data['epoch_accumulate_loss_spa'] = Training_losses_data['epoch_accumulate_loss_spa'] + loss_spa.item() # epoch_accumulate_loss_spa stores the accumulate loss_spa in an epoch, for record purpose only (not for backpropagation)
			Training_losses_data['epoch_average_loss_spa'] = Training_losses_data['epoch_accumulate_loss_spa']/(iteration+1)
			Training_losses_data['epoch_accumulate_loss_col'] = Training_losses_data['epoch_accumulate_loss_col'] + loss_col.item() # epoch_accumulate_loss_col stores the accumulate loss_col in an epoch, for record purpose only (not for backpropagation)
			Training_losses_data['epoch_average_loss_col'] = Training_losses_data['epoch_accumulate_loss_col']/(iteration+1)
			Training_losses_data['epoch_accumulate_loss_exp'] = Training_losses_data['epoch_accumulate_loss_exp'] + loss_exp.item() # epoch_accumulate_loss_exp stores the accumulate loss_exp in an epoch, for record purpose only (not for backpropagation)
			Training_losses_data['epoch_average_loss_exp'] = Training_losses_data['epoch_accumulate_loss_exp']/(iteration+1)
			batch_training_sample_size = LL_image.shape[0] # Get the total number of input samples (LL_image.shape[0]) that have been processed by the model in the current iteration, can be equivalent to batch size.
			Training_losses_data['epoch_accumulate_number_of_training_input_samples_processed'] += batch_training_sample_size # Update the processed input samples in the current iteration. So that at the end, Training_losses_data['epoch_accumulate_number_of_training_input_samples_processed'] stores the total number of input samples that have been processed by the model.



			optimizer.zero_grad() # Remove all previously calculated partial derivatives
			loss.backward() # Perform backpropagation
			torch.nn.utils.clip_grad_norm_(DCE_net.parameters(),config.grad_clip_norm) # Perform Gradient Clipping by Norm to prevent the gradients from becoming excessively large during the training of neural networks, which will lead to exploding gradients problem.
			optimizer.step() # Update the parameters (weights and biases) of the model

			train_bar.set_description_str('Iteration: {}/{} | AccumProcessed_TrainSamples: {} | lr: {:.6f} | EpochAve_loss: {:.4f}'
					.format(iteration + 1, train_number,
			 				Training_losses_data['epoch_accumulate_number_of_training_input_samples_processed'],
							optimizer.param_groups[0]['lr'], 
							Training_losses_data['epoch_average_loss']
						)
					)
			
		scheduler.step()

		if (epoch == 0): # if it reaches the first epoch
			with open(csv_TrainingLossesResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
				writer = csv.DictWriter(csvfile, fieldnames=Training_losses_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Training_losses_data.keys() as the elements=keys of the header
				writer.writeheader() # The writer writes the header on the csv file
					
		with open(csv_TrainingLossesResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_IQA_metrics_data dictionary to that csv file.
			writer = csv.DictWriter(csvfile, fieldnames=Training_losses_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_IQA_metrics_data.keys() as the elements=keys of the header
			writer.writerow(Training_losses_data) # The writer writes the data (value) of Validation_IQA_metrics_data dictionary in sequence as a row on the csv file

		
		# save model parameters at certain checkpoints
		# if epoch != 0:
		if ((epoch+1) % config.snapshot_iter) == 0: # at every (snapshot_iter)th iteration
			save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=0)	

		# -------------------------------------------------------------------
		### **Part 4: Validation**    

		print('Epoch: {}/{} | Model validation begins'.format(Validation_IQA_metrics_data['epoch'] + 1, config.num_epochs))
        
		DCE_net.eval()

		val_bar = tqdm(val_loader)

		
		max_firstNBatches_ImageGroups_VisualizationSample = config.BatchesNum_ImageGroups_VisualizationSample
		if len(val_loader) < max_firstNBatches_ImageGroups_VisualizationSample: # len(val_loader) returns the number of batches of the dataset = total number of iterations/batches. len(val_loader.dataset) returns the number samples in the dataset = the number of input-output-GroundTruth image groups.
			max_firstNBatches_ImageGroups_VisualizationSample = len(val_loader) # when the total batches number of the dataset is less than the specified maximum batches number for visualization, takes the total batches number of the dataset as the maximum batches number for visualization

		save_image = None

		with torch.no_grad():
			for iteration, (val_ori_image, val_LL_image) in enumerate(val_bar):
				
				val_ori_image, val_LL_image = val_ori_image.cuda(), val_LL_image.cuda()
				# print('Val: ori_image.shape():', val_ori_image.shape) # show the shape of validation images
				
				# **Subpart 8: Perform image enhancement on the current batch of samples using the model**
				start = time.time() # Get the starting time of image enhancement process on a batch of samples, in the unit of second.

				# _,enhanced_image,_ = DCE_net(val_LL_image) # Perform image enhancement using Zero-DCE on the current batch of images/samples. enhanced_image stores the enhanced images.

				val_enhanced_image_1,val_enhanced_image,val_A = DCE_net(val_LL_image) # Call the model to enhance this batch of samples, by returning each image enhanced at 4th enhancing iteration, each image enhanced at last enhancing iteration, and the 24 curve parameter maps used to enhance this batch of samples at different stages respectively.
				
				val_batch_duration = (time.time() - start) # Get the duration of image enhancement process on the current batch of samples

				val_loss_TV = config.weightage_loss_TV*L_TV(val_A) # Calculate the illumination smoothness loss of this batch of samples
				val_loss_spa = config.weightage_loss_spa*torch.mean(L_spa(val_enhanced_image, val_LL_image)) # Calculate the spatial consistency loss of this batch of samples
				val_loss_col = config.weightage_loss_col*torch.mean(L_color(val_enhanced_image)) # Calculate the color constancy loss of this batch of samples
				val_loss_exp = config.weightage_loss_exp*torch.mean(L_exp(val_enhanced_image)) # Calculate the exposure control loss of this batch of samples
				# Total loss of this batch of samples (best_loss, considering the weightage of each loss contributed to the total loss)
				val_loss =  val_loss_TV + val_loss_spa + val_loss_col + val_loss_exp 

				# The epoch losses for record purpose only
				Validation_losses_data['epoch_accumulate_loss'] = Validation_losses_data['epoch_accumulate_loss'] + val_loss.item() # epoch_accumulate_loss stores the accumulate loss in an epoch, for record purpose only (not for backpropagation)
				Validation_losses_data['epoch_average_loss'] = Validation_losses_data['epoch_accumulate_loss']/(iteration+1)
				Validation_losses_data['epoch_accumulate_Loss_TV'] = Validation_losses_data['epoch_accumulate_Loss_TV'] + val_loss_TV.item() # epoch_accumulate_Loss_TV stores the accumulate Loss_TV in an epoch, for record purpose only (not for backpropagation)
				Validation_losses_data['epoch_average_Loss_TV'] = Validation_losses_data['epoch_accumulate_Loss_TV']/(iteration+1)
				Validation_losses_data['epoch_accumulate_loss_spa'] = Validation_losses_data['epoch_accumulate_loss_spa'] + val_loss_spa.item() # epoch_accumulate_loss_spa stores the accumulate loss_spa in an epoch, for record purpose only (not for backpropagation)
				Validation_losses_data['epoch_average_loss_spa'] = Validation_losses_data['epoch_accumulate_loss_spa']/(iteration+1)
				Validation_losses_data['epoch_accumulate_loss_col'] = Validation_losses_data['epoch_accumulate_loss_col'] + val_loss_col.item() # epoch_accumulate_loss_col stores the accumulate loss_col in an epoch, for record purpose only (not for backpropagation)
				Validation_losses_data['epoch_average_loss_col'] = Validation_losses_data['epoch_accumulate_loss_col']/(iteration+1)
				Validation_losses_data['epoch_accumulate_loss_exp'] = Validation_losses_data['epoch_accumulate_loss_exp'] + val_loss_exp.item() # epoch_accumulate_loss_exp stores the accumulate loss_exp in an epoch, for record purpose only (not for backpropagation)
				Validation_losses_data['epoch_average_loss_exp'] = Validation_losses_data['epoch_accumulate_loss_exp']/(iteration+1)
				batch_validation_sample_size = val_LL_image.shape[0] # Get the total number of input samples (LL_image.shape[0]) that have been processed by the model in the current iteration, can be equivalent to batch size.
				Validation_losses_data['epoch_accumulate_number_of_training_input_samples_processed'] += batch_validation_sample_size # Update the processed input samples in the current iteration. So that at the end, Training_losses_data['epoch_accumulate_number_of_training_input_samples_processed'] stores the total number of input samples that have been processed by the model.




				# Validation_IQA_metrics_data['batch_duration'] = batch_duration # Update the current batch duration to the Validation_IQA_metrics_data dictionary
				# Validation_ComputationalComplexity_metrics_data['accumulate_batch_duration'] += Validation_IQA_metrics_data['batch_duration'] # Update the accumulate batch duration to the Validation_ComputationalComplexity_metrics_data dictionary
				Validation_ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteOperations'] += val_batch_duration # Update the accumulate batch duration to the Validation_ComputationalComplexity_metrics_data dictionary
				# print("\nDuration of image enhancement process on the current batch of input samples [second (s)]:", Validation_IQA_metrics_data['batch_duration'])
				
				# **Subpart 9: Perform Image Quality Assessment (IQA) metric calculations**
				IQA_metrics_calculation(Validation_IQA_metrics_data) 

				if (epoch == config.num_epochs - 1): # if it reaches the last epoch 
					# **Subpart 12: Perform computation complexity metric calculations**
					ComputationComplexity_metrics_calculation(Validation_ComputationalComplexity_metrics_data)

				if iteration <= (max_firstNBatches_ImageGroups_VisualizationSample - 1):   # before reaching the first max_step number of iterations/batches in each epoch, the val_LL_image (Input), val_enhanced_image (Output), and val_ori_image (GroundTruth) image groups the network has dealt with will be concatenated horizontally together as an image grid. So max_step determines the number of groups will be concatenated horizontally in the image grid. In other words, only the first max_step groups will be chosen as the samples to be concatenated as the image grid to show/visualize the network performance.
					sv_im = torchvision.utils.make_grid(torch.cat((val_LL_image, val_enhanced_image, val_ori_image), 0), nrow=val_ori_image.shape[0])
					if save_image == None:
						save_image = sv_im
					else:
						save_image = torch.cat((save_image, sv_im), dim=2)
				if iteration == (max_firstNBatches_ImageGroups_VisualizationSample - 1):   # when reaching max_step number of iterations/batches in each epoch, the image grid will be saved on the device. The number of image groups in the image grid = firstNBatches * batch_size during validation.
					torchvision.utils.save_image(
						save_image,
						os.path.join(sample_output_folder, '{}-{}-{}-{}-{}-SampleValOutput.jpg'.format(current_date_time_string, config.model_name, config.dataset_name, 'Epoch', epoch))
					)

				val_bar.set_description_str('Iteration: %d/%d | AccumProcessed_ValSamples: %d | EpochAve_loss: %.4f; EpochAve_IQAScore: %.4f; EpochAve_PSNR: %.4f dB; EpochAve_SSIM: %.4f; EpochAve_MAE: %.4f; EpochAve_LPIPS: %.4f' % (
							iteration + 1, val_number, 
							Validation_IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'], 
							Validation_losses_data['epoch_average_loss'], Validation_IQA_metrics_data['epoch_average_IQAScore'], Validation_IQA_metrics_data['epoch_average_psnr_db'], Validation_IQA_metrics_data['epoch_average_ssim'], Validation_IQA_metrics_data['epoch_average_mae'], Validation_IQA_metrics_data['epoch_average_lpips']))
				
			# **Subpart 10: Record the calculated (IQA) and losses metrics on validation set to the respective csv file**
			if (epoch == 0): # if it reaches the first epoch
				# Record calculated IQA metrics on validation set
				with open(csv_ValidationIQAResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
					writer = csv.DictWriter(csvfile, fieldnames=Validation_IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_IQA_metrics_data.keys() as the elements=keys of the header
					writer.writeheader() # The writer writes the header on the csv file
				# Record calculated losses metrics on validation set
				with open(csv_ValidationLossesResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
					writer = csv.DictWriter(csvfile, fieldnames=Validation_losses_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Training_losses_data.keys() as the elements=keys of the header
					writer.writeheader() # The writer writes the header on the csv file
			
			# Record calculated IQA metrics on validation set
			with open(csv_ValidationIQAResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_IQA_metrics_data dictionary to that csv file.
				writer = csv.DictWriter(csvfile, fieldnames=Validation_IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_IQA_metrics_data.keys() as the elements=keys of the header
				writer.writerow(Validation_IQA_metrics_data) # The writer writes the data (value) of Validation_IQA_metrics_data dictionary in sequence as a row on the csv file
			# Record calculated losses metrics on validation set
			with open(csv_ValidationLossesResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_IQA_metrics_data dictionary to that csv file.
				writer = csv.DictWriter(csvfile, fieldnames=Validation_losses_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_IQA_metrics_data.keys() as the elements=keys of the header
				writer.writerow(Validation_losses_data) # The writer writes the data (value) of Validation_IQA_metrics_data dictionary in sequence as a row on the csv file
			
			
			if (epoch == 0): # if it reaches the first epoch				
				best_epoch_validation_average_psnr_db = Validation_IQA_metrics_data['epoch_average_psnr_db']
				best_epoch_validation_average_ssim = Validation_IQA_metrics_data['epoch_average_ssim']
				best_epoch_validation_average_mae = Validation_IQA_metrics_data['epoch_average_mae']
				best_epoch_validation_average_lpips = Validation_IQA_metrics_data['epoch_average_lpips']
				best_epoch_validation_average_loss = Validation_losses_data['epoch_average_loss']
				best_epoch_validation_IQAScore = Validation_IQA_metrics_data['epoch_average_IQAScore']

			# Save the checkpoint that has the best IQA results respectively
			if  Validation_IQA_metrics_data['epoch_average_psnr_db'] > best_epoch_validation_average_psnr_db :
				best_epoch_validation_average_psnr_db = Validation_IQA_metrics_data['epoch_average_psnr_db']
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=1)	
				
			if  Validation_IQA_metrics_data['epoch_average_ssim'] > best_epoch_validation_average_ssim :
				best_epoch_validation_average_ssim = Validation_IQA_metrics_data['epoch_average_ssim']
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=2)

			if  Validation_IQA_metrics_data['epoch_average_mae'] < best_epoch_validation_average_mae :
				best_epoch_validation_average_mae = Validation_IQA_metrics_data['epoch_average_mae']
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=3)	
			
			if  Validation_IQA_metrics_data['epoch_average_lpips'] < best_epoch_validation_average_lpips :
				best_epoch_validation_average_lpips = Validation_IQA_metrics_data['epoch_average_lpips']
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=4)

			if  Validation_losses_data['epoch_average_loss'] < best_epoch_validation_average_loss :
				best_epoch_validation_average_loss = Validation_losses_data['epoch_average_loss']
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=5)

			if  Validation_IQA_metrics_data['epoch_average_IQAScore'] > best_epoch_validation_IQAScore :
				best_epoch_validation_IQAScore = Validation_IQA_metrics_data['epoch_average_IQAScore']
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=6)

			if (epoch == config.num_epochs - 1): # if it reaches the last epoch 
				save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, scheduler, key=7)


			# Early stopping section:
			if config.earlystopping_switch:
				# If epoch_average_loss on validation set and all IQA metrics on validation set do not improve for 'patience' epochs, stop training early.
				if (
						Validation_IQA_metrics_data['epoch_average_psnr_db'] <= best_epoch_validation_average_psnr_db  and \
						Validation_IQA_metrics_data['epoch_average_ssim'] <= best_epoch_validation_average_ssim   and \
						Validation_IQA_metrics_data['epoch_average_mae'] >= best_epoch_validation_average_mae     and \
						Validation_IQA_metrics_data['epoch_average_lpips'] >= best_epoch_validation_average_lpips and \
						Validation_IQA_metrics_data['epoch_average_IQAScore'] <= best_epoch_validation_IQAScore and \
						Validation_losses_data['epoch_average_loss'] >= best_epoch_validation_average_loss
					):

					wait += 1
					if wait >= patience:
						stopped_epoch = epoch
						print("Early stopped training at epoch %d" % stopped_epoch)

						ComputationComplexity_metrics_calculation(Validation_ComputationalComplexity_metrics_data)
						
						break
				else:
					wait = 0

				

		DCE_net.train()

# **Subpart 13: Record the calculated computation complexity metrics to that csv file**
	with open(csv_ValidationIQAResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of Validation_ComputationalComplexity_metrics_data dictionary to that csv file.
		writer = csv.DictWriter(csvfile, fieldnames=Validation_ComputationalComplexity_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Validation_ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
		writer.writeheader() # The writer writes the header on the csv file
		writer.writerow(Validation_ComputationalComplexity_metrics_data)  # The writer writes the data (value) of Validation_ComputationalComplexity_metrics_data dictionary in sequence as a row on the csv file

	# -------------------------------------------------------------------
    # train finish

	generate_save_History() # generate and save the results history as images
	

	print("-------Operations completed-------")

	
	



# Task: Make a copy of this script, then->
# 1) [Done] At each epoch, try use reduction = sum to directly add psnr_subtotal of each batch, then at last batch only divide the total psnr_subtotal of all batches to get the average psnr of that epoch. Then verify the results by checking the psnr of each image, by using reduction = 'none'. More info: https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/utilities/distributed.py#L22
# 2) [Done] Verify all IQA metrics and Computational Complexity metrics calculation
# 3) [Done] Add csv to record the average total loss at each epoch, then plot its graph over epoch and save on device. Same goes to IQA metrics.
# 4) Summary: Take this script as the template to perform model training. This is the best version out of all versions tried.






