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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

	
def IQA_metrics_calculation(IQA_metrics_data): # We define IQA_metrics_data as the input argument, so that the results related to IQA_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	sample_size = LL_image.shape[0] # Get the total number of input samples (img_lowlight.shape[0]) that have been processed by the model in the current iteration, can be equivalent to batch size.
	IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] += sample_size # Update the processed input samples in the current iteration. So that at the end, IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] stores the total number of input samples that have been processed by the model.

	# PSNR part
	batch_sum_psnr = PSNR(enhanced_image, ori_image) # get the total PSNR of a batch of images
	IQA_metrics_data['epoch_accumulate_psnr'] += batch_sum_psnr
	IQA_metrics_data['epoch_average_psnr'] = IQA_metrics_data['epoch_accumulate_psnr'] / IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] # Get the average PSNR of all image pairs the model has gone through. The mathematical concept behinds is: (Summation of PSNR of all image pairs the model has gone through)/[Total batches of images the model has gone through, which is same as the total number of image pairs the model has gone through]).

	# SSIM part
	batch_average_ssim = ssim(enhanced_image, ori_image).item()
	IQA_metrics_data['epoch_accumulate_ssim'] += batch_average_ssim * sample_size
	IQA_metrics_data['epoch_average_ssim'] = IQA_metrics_data['epoch_accumulate_ssim'] / IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed']

	# MAE part
	batch_average_mae = mae(enhanced_image, ori_image).item()
	IQA_metrics_data['epoch_accumulate_mae'] += batch_average_mae * sample_size
	IQA_metrics_data['epoch_average_mae'] = IQA_metrics_data['epoch_accumulate_mae'] / IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed']
	
	# LPIPS part
	batch_average_lpips = lpips(enhanced_image, ori_image).item()
	IQA_metrics_data['epoch_accumulate_lpips'] += batch_average_lpips * sample_size
	IQA_metrics_data['epoch_average_lpips'] = IQA_metrics_data['epoch_accumulate_lpips'] / IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed']

def ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data): # We define ComputationalComplexity_metrics_data as the input argument, so that the results related to ComputationalComplexity_metrics_data calculated in this self-defined function can be directly updated to the metric dictionary located in the main() of this script.
	
	# Calculate the average runtime using the total batch duration and total number of input samples (means the average runtime for an input sample)
	ComputationalComplexity_metrics_data['average_val_runtime_forCompleteOperations'] = ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteOperations'] / (IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'] * config.num_epochs)
	
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

def save_model(epoch, path, net, optimizer, net_name, key): # this function saves model checkpoints
	if not os.path.exists(os.path.join(path, net_name)): # if the folder that stores the checkpoints does not exist
		os.makedirs(os.path.join(path, net_name))

	if key == 0:
		# torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') # The format used by the author. Save the parameters (weigths and biases) of the model at the specified snapshot (EG:epoch) as a pth file	
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, # save the checkpoints for each config.snap_iter interval
					f=os.path.join(path, net_name, '{}-{}-{}-{}-{}-checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name, 'Epoch', epoch)))
	elif key == 1:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, # save the checkpoints that has the best epoch_average_psnr
					f=os.path.join(path, net_name, '{}-{}-{}-BestEpochAveragePSNR_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)))

	elif key == 2:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, # save the checkpoints that has the best epoch_average_ssim
					f=os.path.join(path, net_name, '{}-{}-{}-BestEpochAverageSSIM_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)))

	elif key == 3:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, # save the checkpoints that has the best epoch_average_mae
					f=os.path.join(path, net_name, '{}-{}-{}-BestEpochAverageMAE_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)))

	elif key == 4:
		torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, # save the checkpoints that has the best epoch_average_lpips
					f=os.path.join(path, net_name, '{}-{}-{}-BestEpochAverageLPIPS_checkpoint.pt'.format(current_date_time_string, config.model_name, config.dataset_name)))


def generate_save_TrainingResults_History():
	

	fig = plt.figure(figsize=(10, 10), dpi=100, constrained_layout=True)
	fig.suptitle('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results: Losses') # set the figure super title
	gs = fig.add_gridspec(3, 2)

	# (Training results) Generate and save the figure of [Average loss vs Epoch]
	epoch_training_average_loss_list_ymin = min(epoch_training_average_loss_list)
	epoch_training_average_loss_list_xpos = epoch_training_average_loss_list.index(epoch_training_average_loss_list_ymin)
	epoch_training_average_loss_list_xmin = epoch_list[epoch_training_average_loss_list_xpos]
	ax1 = fig.add_subplot(gs[0, :])
	ax1.plot(epoch_list, epoch_training_average_loss_list, 'r') #row=0, col=0, 1
	ax1.plot(epoch_training_average_loss_list_xmin, epoch_training_average_loss_list_ymin, 'r', marker='o', fillstyle='none') # plot the minimum point
	ax1.set_ylabel('Average loss') # set the y-label
	ax1.set_xlabel('Epoch') # set the x-label
	ax1.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax1.set_title(f'Y-min. coord.:[{epoch_training_average_loss_list_xmin},{epoch_training_average_loss_list_ymin:.4f}]')

	# (Training results) Generate and save the figure of [Average loss_TV vs Epoch]
	epoch_training_average_loss_TV_list_ymin = min(epoch_training_average_loss_TV_list)
	epoch_training_average_loss_TV_list_xpos = epoch_training_average_loss_TV_list.index(epoch_training_average_loss_TV_list_ymin)
	epoch_training_average_loss_TV_list_xmin = epoch_list[epoch_training_average_loss_TV_list_xpos]
	ax2 = fig.add_subplot(gs[1, 0])
	ax2.plot(epoch_list, epoch_training_average_loss_TV_list, 'b') # plot the graph
	ax2.plot(epoch_training_average_loss_TV_list_xmin, epoch_training_average_loss_TV_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax2.set_ylabel('Average loss_TV') # set the y-label
	ax2.set_xlabel('Epoch') # set the x-label
	ax2.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax2.set_title(f'Y-min. coord.:[{epoch_training_average_loss_TV_list_xmin},{epoch_training_average_loss_TV_list_ymin:.4f}]')

	# (Training results) Generate and save the figure of [Average loss_spa vs Epoch]
	epoch_training_average_loss_spa_list_ymin = min(epoch_training_average_loss_spa_list)
	epoch_training_average_loss_spa_list_xpos = epoch_training_average_loss_spa_list.index(epoch_training_average_loss_spa_list_ymin)
	epoch_training_average_loss_spa_list_xmin = epoch_list[epoch_training_average_loss_spa_list_xpos]
	ax3 = fig.add_subplot(gs[1, 1])
	ax3.plot(epoch_list, epoch_training_average_loss_spa_list) # plot the graph
	ax3.plot(epoch_training_average_loss_spa_list_xmin, epoch_training_average_loss_spa_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax3.set_ylabel('Average loss_spa') # set the y-label
	ax3.set_xlabel('Epoch') # set the x-label
	ax3.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax3.set_title(f'Y-min. coord.:[{epoch_training_average_loss_spa_list_xmin},{epoch_training_average_loss_spa_list_ymin:.4f}]')

	# (Training results) Generate and save the figure of [Average loss_col vs Epoch]
	epoch_training_average_loss_col_list_ymin = min(epoch_training_average_loss_col_list)
	epoch_training_average_loss_col_list_xpos = epoch_training_average_loss_col_list.index(epoch_training_average_loss_col_list_ymin)
	epoch_training_average_loss_col_list_xmin = epoch_list[epoch_training_average_loss_col_list_xpos]
	ax4 = fig.add_subplot(gs[2, 0])
	ax4.plot(epoch_list, epoch_training_average_loss_col_list) # plot the graph
	ax4.plot(epoch_training_average_loss_col_list_xmin, epoch_training_average_loss_col_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax4.set_ylabel('Average loss_col') # set the y-label
	ax4.set_xlabel('Epoch') # set the x-label
	ax4.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax4.set_title(f'Y-min. coord.:[{epoch_training_average_loss_col_list_xmin},{epoch_training_average_loss_col_list_ymin:.4f}]')

	# (Training results) Generate and save the figure of [Average loss_exp vs Epoch]
	epoch_training_average_loss_exp_list_ymin = min(epoch_training_average_loss_exp_list)
	epoch_training_average_loss_exp_list_xpos = epoch_training_average_loss_exp_list.index(epoch_training_average_loss_exp_list_ymin)
	epoch_training_average_loss_exp_list_xmin = epoch_list[epoch_training_average_loss_exp_list_xpos]
	ax5 = fig.add_subplot(gs[2, 1])
	ax5.plot(epoch_list, epoch_training_average_loss_exp_list) # plot the graph
	ax5.plot(epoch_training_average_loss_exp_list_xmin, epoch_training_average_loss_exp_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax5.set_ylabel('Average loss_exp') # set the y-label
	ax5.set_xlabel('Epoch') # set the x-label
	ax5.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax5.set_title(f'Y-min. coord.:[{epoch_training_average_loss_exp_list_xmin},{epoch_training_average_loss_exp_list_ymin:.4f}]')

	epoch_training_results_history_filename = '{}-{}-{}-epoch_training_results_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	epoch_training_results_history_filepath = os.path.join(resultPath_csv, epoch_training_results_history_filename) # define the filepath, used to save the figure as an image
	
	plt.margins() 
	plt.savefig(epoch_training_results_history_filepath, bbox_inches='tight') # save the figure as an image
	plt.show()

	# # (Training results) Generate and save the figure of [Average loss vs Epoch] **DONE
	# plt.figure() # creates a new figure
	print('epoch_training_average_loss_list:', epoch_training_average_loss_list)
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
	print('epoch_training_average_loss_TV_list:', epoch_training_average_loss_TV_list)
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
	print('epoch_training_average_loss_spa_list:', epoch_training_average_loss_spa_list)
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
	print('epoch_training_average_loss_col_list:', epoch_training_average_loss_col_list)
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
	print('epoch_training_average_loss_exp_list:', epoch_training_average_loss_exp_list)
	# x_epoch_training_average_loss_exp_list = [x for x in range(len(epoch_training_average_loss_exp_list))] # create the x-axis elements
	# plt.plot(x_epoch_training_average_loss_exp_list, epoch_training_average_loss_exp_list) # plot the graph
	# plt.xlabel('Epoch') # set the x-label
	# plt.ylabel('Average loss_exp') # set the y-label
	# plt.title('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nTraining Results [Average loss_exp vs Epoch]') # set the figure title
	# plt.xticks(np.arange(min(x_epoch_training_average_loss_exp_list), max(x_epoch_training_average_loss_exp_list)+1, 1)) # set the interval of x-axis 
	# epoch_training_average_loss_exp_history_filename = '{}-{}-{}-epoch_training_average_loss_exp_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	# epoch_training_average_loss_exp_history_filepath = os.path.join(resultPath_csv, epoch_training_average_loss_exp_history_filename) # define the filepath, used to save the figure as an image
	# plt.savefig(epoch_training_average_loss_exp_history_filepath, bbox_inches='tight') # save the figure as an image
	

def generate_save_ValidationResults_History():

	fig = plt.figure(figsize=(10, 10), dpi=100, constrained_layout=True)
	fig.suptitle('{}-{}-{}'.format(current_date_time_string, config.model_name, config.dataset_name) + '\nValidation Results: IQA Metrics') # set the figure super title
	gs = fig.add_gridspec(2, 2) # create 2x2 grid

	# (Validation results) Generate and save the figure of [Average PSNR vs Epoch]
	epoch_validation_average_psnr_list_ymax = max(epoch_validation_average_psnr_list)
	epoch_validation_average_psnr_list_xpos = epoch_validation_average_psnr_list.index(epoch_validation_average_psnr_list_ymax)
	epoch_validation_average_psnr_list_xmax = epoch_list[epoch_validation_average_psnr_list_xpos]
	ax6 = fig.add_subplot(gs[0, 0])
	ax6.plot(epoch_list, epoch_validation_average_psnr_list, 'b') #row=0, col=0, 1
	ax6.plot(epoch_validation_average_psnr_list_xmax, epoch_validation_average_psnr_list_ymax, 'b', marker='o', fillstyle='none') # plot the maximum point
	ax6.set_ylabel('Average PSNR [dB]') # set the y-label
	ax6.set_xlabel('Epoch') # set the x-label
	ax6.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax6.set_title(f'Y-max. coord.:[{epoch_validation_average_psnr_list_xmax},{epoch_validation_average_psnr_list_ymax:.4f}]')

	# (Validation results) Generate and save the figure of [Average SSIM vs Epoch]
	epoch_validation_average_ssim_list_ymax = max(epoch_validation_average_ssim_list)
	epoch_validation_average_ssim_list_xpos = epoch_validation_average_ssim_list.index(epoch_validation_average_ssim_list_ymax)
	epoch_validation_average_ssim_list_xmax = epoch_list[epoch_validation_average_ssim_list_xpos]
	ax7 = fig.add_subplot(gs[0, 1])
	ax7.plot(epoch_list, epoch_validation_average_ssim_list, 'b') #row=0, col=0, 1
	ax7.plot(epoch_validation_average_ssim_list_xmax, epoch_validation_average_ssim_list_ymax, 'b', marker='o', fillstyle='none') # plot the maximum point
	ax7.set_ylabel('Average SSIM') # set the y-label
	ax7.set_xlabel('Epoch') # set the x-label
	ax7.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax7.set_title(f'Y-max. coord.:[{epoch_validation_average_ssim_list_xmax},{epoch_validation_average_ssim_list_ymax:.4f}]')

	# (Validation results) Generate and save the figure of [Average MAE vs Epoch]
	epoch_validation_average_mae_list_ymin = min(epoch_validation_average_mae_list)
	epoch_validation_average_mae_list_xpos = epoch_validation_average_mae_list.index(epoch_validation_average_mae_list_ymin)
	epoch_validation_average_mae_list_xmin = epoch_list[epoch_validation_average_mae_list_xpos]
	ax8 = fig.add_subplot(gs[1, 0])
	ax8.plot(epoch_list, epoch_validation_average_mae_list, 'b') #row=0, col=0, 1
	ax8.plot(epoch_validation_average_mae_list_xmin, epoch_validation_average_mae_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax8.set_ylabel('Average MAE') # set the y-label
	ax8.set_xlabel('Epoch') # set the x-label
	ax8.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax8.set_title(f'Y-min. coord.:[{epoch_validation_average_mae_list_xmin},{epoch_validation_average_mae_list_ymin:.4f}]')


	# (Validation results) Generate and save the figure of [Average LPIPS vs Epoch]
	epoch_validation_average_lpips_list_ymin = min(epoch_validation_average_lpips_list)
	epoch_validation_average_lpips_list_xpos = epoch_validation_average_lpips_list.index(epoch_validation_average_lpips_list_ymin)
	epoch_validation_average_lpips_list_xmin = epoch_list[epoch_validation_average_lpips_list_xpos]
	ax9 = fig.add_subplot(gs[1, 1])
	ax9.plot(epoch_list, epoch_validation_average_lpips_list, 'b') #row=0, col=0, 1
	ax9.plot(epoch_validation_average_lpips_list_xmin, epoch_validation_average_lpips_list_ymin, 'b', marker='o', fillstyle='none') # plot the minimum point
	ax9.set_ylabel('Average LPIPS') # set the y-label
	ax9.set_xlabel('Epoch') # set the x-label
	ax9.set_xticks(np.arange(min(epoch_list), max(epoch_list)+1, config.snapshot_iter)) # set the interval of x-axis 
	ax9.set_title(f'Y-min. coord.:[{epoch_validation_average_lpips_list_xmin},{epoch_validation_average_lpips_list_ymin:.4f}]')


	epoch_validation_results_filename = '{}-{}-{}-epoch_validation_results_history.jpg'.format(current_date_time_string, config.model_name, config.dataset_name) # define the filename
	epoch_validation_results_filepath = os.path.join(resultPath_csv, epoch_validation_results_filename) # define the filepath, used to save the figure as an image

	plt.margins() 
	plt.savefig(epoch_validation_results_filepath, bbox_inches='tight') # save the figure as an image
	plt.show()

	# # (Validation results) Generate and save the figure of [Average PSNR vs Epoch] **DONE
	# plt.figure() # creates a new figure
	print('epoch_validation_average_psnr_list:', epoch_validation_average_psnr_list)
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
	print('epoch_validation_average_ssim_list:', epoch_validation_average_ssim_list)
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
	print('epoch_validation_average_mae_list:', epoch_validation_average_mae_list)
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
	print('epoch_validation_average_lpips_list:', epoch_validation_average_lpips_list)
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
	parser.add_argument('--val_batch_size', type=int, default=3)
	parser.add_argument('--num_workers', type=int, default=4)
	# parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=1)
	parser.add_argument('--snapshots_folder', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	# parser.add_argument('--pretrain_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/snapshots/Epoch99.pth") # The pretrained model parameters provided by the author
	parser.add_argument('--checkpoint_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/results/model_parameters/Zero-DCE/enhance_1.pt") # The pretrained model parameters obtained by myself
	parser.add_argument('--BatchesNum_ImageGroups_VisualizationSample', type=int, default=20) # At each epoch, save the maximum first N batches of LL_image (Input), enhanced_image (Output), and ori_image (GroundTruth) image groups in validation as the samples to visualize the network performance 
	parser.add_argument('--earlystopping_patience', type=int, default = 5) # the threshold (the number of consecutive epoch for no improvement on all IQA metrics) to wait before stopping model training, when there are consecutives no improvements on all IQA metrics

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
	

	# For the folder that stores the checkpoints containing the states of epoch, model parameters, optimizer, and loss [Used to apply on model for inference/resume training]
	resultPath_subfolder_modelparameters = "model_parameters" # Define the folder that stores the grids of input-output image pairs involved in metric calculations
	resultPath_ModelParametersResults = os.path.join(resultPath, resultPath_subfolder_modelparameters) # Define the absolute path to the folder that stores the grids of input-output image pairs involved in metric calculations
	os.makedirs(resultPath_ModelParametersResults, exist_ok=True) # Ensure the path that stores the grids of input-output image pairs involved in metric calculations is created
	
	# For the folder that stores the training and validation results history
	# For the csv that stores the metrics data
	resultPath_subfolder_csv = "csvFile" # Define the folder that stores the csv file
	resultPath_csv = os.path.join(resultPath, resultPath_subfolder_csv) # Define the absolute path to the folder that stores the csv file
	os.makedirs(resultPath_csv, exist_ok=True) # Ensure the path that stores the csv file is created
	current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)

	# To record training results 
	csv_TrainingResult_filename = current_date_time_string  + '-' + config.model_name + '-' + config.dataset_name + '-LLIE-TrainingResults-History' + '.csv' # Create the filename of the csv that stores the metrics data
	csv_TrainingResult_filepath = os.path.join(resultPath_csv, csv_TrainingResult_filename) # Create the path to the csv that stores the metrics data

	# To record validation results 
	csv_ValidationResult_filename = current_date_time_string  + '-' + config.model_name + '-' + config.dataset_name + '-LLIE-ValidationResults-History' + '.csv' # Create the filename of the csv that stores the metrics data
	csv_ValidationResult_filepath = os.path.join(resultPath_csv, csv_ValidationResult_filename) # Create the path to the csv that stores the metrics data
	# with open(csv_ValidationResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
	# 	writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
	# 	writer.writeheader() # The writer writes the header on the csv file

	# For the folder that stores the grids of input-output-GroundTruth image groups involved in metric calculations
	sample_output_folder = os.path.join(resultPath,'sample_output_folder') # Create a new path by changing a part of resultPath
	sample_dir = os.path.join(sample_output_folder, config.model_name)
	if not os.path.isdir(sample_dir):
		os.makedirs(sample_dir)

# **Subpart 2: Initialize training and validation dataset**
	train_loader, train_number, val_loader, val_number = load_data(config)


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
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1) # Decays the learning rate of each parameter group by gamma every step_size epochs.

# **Subpart 6: Initialize the IQA metric-related functions to calculate IQA results during validation**
	ssim = SSIM(data_range=1.).cuda() # Performance metric: SSIM
	mae = MAE(num_outputs=1).cuda() # Performance metric: MAE
	lpips = LPIPS(net_type='alex', normalize=True).cuda() # Performance metric: LPIPS (perceptual loss provided by a pretrained learning-based model), using alexnet as the backbone (LPIPS with alexnet performs the best according to its official Github)

# ** Initialize the lists to record different average losses at each epoch of training, for plotting purposes
	epoch_list = []
	epoch_training_average_loss_list = []
	epoch_training_average_loss_TV_list = []
	epoch_training_average_loss_spa_list = []
	epoch_training_average_loss_col_list = []
	epoch_training_average_loss_exp_list = []

# ** Initialize the lists to record different average IQA results at each epoch of validation, for plotting purposes
	epoch_validation_average_psnr_list = []
	epoch_validation_average_ssim_list = []
	epoch_validation_average_mae_list = []
	epoch_validation_average_lpips_list = []

# ** Initialize the variables and threshold related to early stopping
	patience = config.earlystopping_patience # the threshold (the number of consecutive epoch for no improvement on all IQA metrics) to wait before stopping model training, when there are consecutives no improvements on all IQA metrics
	wait = 0
	# current_epoch_validation_average_psnr = torch.tensor([-1])
	# current_epoch_validation_average_psnr = current_epoch_validation_average_psnr.cuda() 
	# current_epoch_validation_average_ssim = torch.tensor([-1])
	# current_epoch_validation_average_ssim = current_epoch_validation_average_ssim.cuda() 
	# current_epoch_validation_average_mae = torch.tensor([-1])
	# current_epoch_validation_average_mae = current_epoch_validation_average_mae.cuda() 
	# current_epoch_validation_average_lpips = torch.tensor([-1])
	# current_epoch_validation_average_lpips = current_epoch_validation_average_lpips.cuda() 
	# best_epoch_validation_average_psnr = torch.tensor([-torch.inf])
	# best_epoch_validation_average_psnr = best_epoch_validation_average_psnr.cuda()
	# best_epoch_validation_average_ssim = torch.tensor([-torch.inf])
	# best_epoch_validation_average_ssim = best_epoch_validation_average_ssim.cuda()
	# best_epoch_validation_average_mae = torch.tensor([torch.inf])
	# best_epoch_validation_average_mae = best_epoch_validation_average_mae.cuda()
	# best_epoch_validation_average_lpips = torch.tensor([torch.inf])
	# best_epoch_validation_average_lpips = best_epoch_validation_average_lpips.cuda()

### **Part 3: Training**
	print('-----Operations begin-----')
	DCE_net.train()

	# **Subpart 2: Initialize a dictionary to store the computational complexity metric data , so they will be updated from time to time later**
	ComputationalComplexity_metrics_data ={'average_val_runtime_forCompleteOperations':0., 'trainable_parameters':0., 'MACs':0., 'FLOPs': 0., 'accumulate_val_batch_duration_forCompleteOperations':0.}
	
	torch.manual_seed(1) # This function ensures the reproducibility of the shuffled indices used in the train_loader and val_loader. The PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch)
	for epoch in range(config.num_epochs):

		# **Subpart 1: Initialize a dictionary to store the Image Quality Assessment (IQA) metric data , so they will be updated from time to time later**
		IQA_metrics_data = {'epoch':0, 'epoch_average_psnr': 0., 'epoch_average_ssim': 0., 'epoch_average_mae': 0., 'epoch_average_lpips': 0., 'epoch_accumulate_number_of_val_input_samples_processed': 0., 'epoch_accumulate_psnr': 0., 'epoch_accumulate_ssim': 0., 'epoch_accumulate_mae': 0.,'epoch_accumulate_lpips': 0.}
		IQA_metrics_data['epoch'] = epoch # Update the current epoch (The number of epoch and batch, when necessary) to the IQA_metrics_data dictionary
		
		Training_losses_data ={'epoch':0, 'epoch_average_loss':0., 'epoch_average_Loss_TV':0., 'epoch_average_loss_spa':0., 'epoch_average_loss_col': 0., 'epoch_average_loss_exp':0., 'epoch_accumulate_number_of_training_input_samples_processed': 0, 'epoch_accumulate_loss':0., 'epoch_accumulate_Loss_TV':0., 'epoch_accumulate_loss_spa':0., 'epoch_accumulate_loss_col': 0., 'epoch_accumulate_loss_exp':0.}
		Training_losses_data['epoch'] = epoch # Update the current epoch (The number of epoch and batch, when necessary) to the IQA_metrics_data dictionary
		
		epoch_list.append(epoch)

		print('Epoch: {}/{} | Model training begins'.format(Training_losses_data['epoch'] + 1, config.num_epochs))
		train_bar = tqdm(train_loader) # tqdm automatically detects the length of the iterable object (train_loader) [The length of the iterable object = The total number of batches of samples] and generates a progress bar that updates dynamically as each item=(batch of samples) is processed.
		for iteration, (ori_image, LL_image) in enumerate(train_bar):
			# count = epoch * train_number + (iteration + 1)
			ori_image, LL_image = ori_image.cuda(), LL_image.cuda() # The model requires normallight-lowlight image pairs as inputs          
			# print('ori_image:', ori_image)
			# print('LL_image:', LL_image)
			enhanced_image_1,enhanced_image,A = DCE_net(LL_image) # Call the model to enhance this batch of samples, by returning each image enhanced at 4th enhancing iteration, each image enhanced at last enhancing iteration, and the 24 curve parameter maps used to enhance this batch of samples at different stages respectively.
				
			loss_TV = 200*L_TV(A) # Calculate the illumination smoothness loss of this batch of samples
			loss_spa = torch.mean(L_spa(enhanced_image, LL_image)) # Calculate the spatial consistency loss of this batch of samples
			loss_col = 5*torch.mean(L_color(enhanced_image)) # Calculate the color constancy loss of this batch of samples
			loss_exp = 10*torch.mean(L_exp(enhanced_image)) # Calculate the exposure control loss of this batch of samples
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

			train_bar.set_description_str('Iteration: {}/{} | Accumulated processed training samples: {} | lr: {:.6f} | Epoch average loss: {:.6f}'
					.format(iteration + 1, train_number,
			 				Training_losses_data['epoch_accumulate_number_of_training_input_samples_processed'],
							optimizer.param_groups[0]['lr'], 
							Training_losses_data['epoch_average_loss']
						)
					)
			
		# scheduler.step()

		if (epoch == 0): # if it reaches the first epoch
			with open(csv_TrainingResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
				writer = csv.DictWriter(csvfile, fieldnames=Training_losses_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and Training_losses_data.keys() as the elements=keys of the header
				writer.writeheader() # The writer writes the header on the csv file
					
		with open(csv_TrainingResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
			writer = csv.DictWriter(csvfile, fieldnames=Training_losses_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
			writer.writerow(Training_losses_data) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file


		epoch_training_average_loss_list.append(Training_losses_data['epoch_average_loss'])
		epoch_training_average_loss_TV_list.append(Training_losses_data['epoch_average_Loss_TV'])
		epoch_training_average_loss_spa_list.append(Training_losses_data['epoch_average_loss_spa'])
		epoch_training_average_loss_col_list.append(Training_losses_data['epoch_average_loss_col'])
		epoch_training_average_loss_exp_list.append(Training_losses_data['epoch_average_loss_exp'])


		# save model parameters at certain checkpoints
		if ((epoch+1) % config.snapshot_iter) == 0: # at every (snapshot_iter)th iteration
			save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, config.model_name, key=0)	

		# -------------------------------------------------------------------
		### **Part 4: Validation**    

		print('Epoch: {}/{} | Model validation begins'.format(IQA_metrics_data['epoch'] + 1, config.num_epochs))
        
		DCE_net.eval()

		val_bar = tqdm(val_loader)

		
		max_firstNBatches_ImageGroups_VisualizationSample = config.BatchesNum_ImageGroups_VisualizationSample
		if len(val_loader) < max_firstNBatches_ImageGroups_VisualizationSample: # len(val_loader) returns the number of batches of the dataset = total number of iterations/batches. len(val_loader.dataset) returns the number samples in the dataset = the number of input-output-GroundTruth image groups.
			max_firstNBatches_ImageGroups_VisualizationSample = len(val_loader) # when the total batches number of the dataset is less than the specified maximum batches number for visualization, takes the total batches number of the dataset as the maximum batches number for visualization

		save_image = None

		with torch.no_grad():
			for iteration, (ori_image, LL_image) in enumerate(val_bar):
				
				ori_image, LL_image = ori_image.cuda(), LL_image.cuda()

				
				# **Subpart 8: Perform image enhancement on the current batch of samples using the model**
				start = time.time() # Get the starting time of image enhancement process on a batch of samples, in the unit of second.

				_,enhanced_image,_ = DCE_net(LL_image) # Perform image enhancement using Zero-DCE on the current batch of images/samples. enhanced_image stores the enhanced images.

				val_batch_duration = (time.time() - start) # Get the duration of image enhancement process on the current batch of samples
				# IQA_metrics_data['batch_duration'] = batch_duration # Update the current batch duration to the IQA_metrics_data dictionary
				# ComputationalComplexity_metrics_data['accumulate_batch_duration'] += IQA_metrics_data['batch_duration'] # Update the accumulate batch duration to the ComputationalComplexity_metrics_data dictionary
				ComputationalComplexity_metrics_data['accumulate_val_batch_duration_forCompleteOperations'] += val_batch_duration # Update the accumulate batch duration to the ComputationalComplexity_metrics_data dictionary
				# print("\nDuration of image enhancement process on the current batch of input samples [second (s)]:", IQA_metrics_data['batch_duration'])
				
				# **Subpart 9: Perform Image Quality Assessment (IQA) metric calculations**
				IQA_metrics_calculation(IQA_metrics_data) 

				if (epoch == config.num_epochs - 1): # if it reaches the last epoch 
					# **Subpart 12: Perform computation complexity metric calculations**
					ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data)

				if iteration <= (max_firstNBatches_ImageGroups_VisualizationSample - 1):   # before reaching the first max_step number of iterations/batches in each epoch, the LL_image (Input), enhanced_image (Output), and ori_image (GroundTruth) image groups the network has dealt with will be concatenated horizontally together as an image grid. So max_step determines the number of groups will be concatenated horizontally in the image grid. In other words, only the first max_step groups will be chosen as the samples to be concatenated as the image grid to show/visualize the network performance.
					sv_im = torchvision.utils.make_grid(torch.cat((LL_image, enhanced_image, ori_image), 0), nrow=ori_image.shape[0])
					if save_image == None:
						save_image = sv_im
					else:
						save_image = torch.cat((save_image, sv_im), dim=2)
				if iteration == (max_firstNBatches_ImageGroups_VisualizationSample - 1):   # when reaching max_step number of iterations/batches in each epoch, the image grid will be saved on the device. The number of image groups in the image grid = firstNBatches * batch_size.
					torchvision.utils.save_image(
						save_image,
						os.path.join(sample_dir, '{}-{}-{}-{}-{}-VisualiationImageGroupsSamples.jpg'.format(current_date_time_string, config.model_name, config.dataset_name, 'Epoch', epoch))
					)

				val_bar.set_description_str('Iteration: %d/%d | Accumulated processed validation samples: %d | Average PSNR: %.4f dB; Average SSIM: %.4f; Average MAE:  %.4f; Average LPIPS: %.4f' % (
							iteration + 1, val_number, 
							IQA_metrics_data['epoch_accumulate_number_of_val_input_samples_processed'], 
							IQA_metrics_data['epoch_average_psnr'], IQA_metrics_data['epoch_average_ssim'], IQA_metrics_data['epoch_average_mae'], IQA_metrics_data['epoch_average_lpips']))
				
			# **Subpart 10: Record the calculated (IQA) metrics to that csv file**
			if (epoch == 0): # if it reaches the first epoch
				with open(csv_ValidationResult_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_ValidationResult_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
					writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
					writer.writeheader() # The writer writes the header on the csv file
							
			with open(csv_ValidationResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
				writer = csv.DictWriter(csvfile, fieldnames=IQA_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
				writer.writerow(IQA_metrics_data) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file

			
			epoch_validation_average_psnr_list.append(IQA_metrics_data['epoch_average_psnr'])
			epoch_validation_average_ssim_list.append(IQA_metrics_data['epoch_average_ssim'])
			epoch_validation_average_mae_list.append(IQA_metrics_data['epoch_average_mae'])
			epoch_validation_average_lpips_list.append(IQA_metrics_data['epoch_average_lpips'])


			# Early stopping section:
			if (epoch == 0): # if it reaches the first epoch
				best_epoch_validation_average_psnr = IQA_metrics_data['epoch_average_psnr']
				best_epoch_validation_average_ssim = IQA_metrics_data['epoch_average_ssim']
				best_epoch_validation_average_mae = IQA_metrics_data['epoch_average_mae']
				best_epoch_validation_average_lpips = IQA_metrics_data['epoch_average_lpips']

			# If all IQA metrics do not improve for 'patience' epochs, stop training early.
			elif (
					IQA_metrics_data['epoch_average_psnr'] <= best_epoch_validation_average_psnr  and \
					IQA_metrics_data['epoch_average_ssim'] <= best_epoch_validation_average_ssim   and \
					IQA_metrics_data['epoch_average_mae'] >= best_epoch_validation_average_mae     and \
					IQA_metrics_data['epoch_average_lpips'] >= best_epoch_validation_average_lpips
				):

				wait += 1
				if wait >= patience:
					stopped_epoch = epoch
					print("Early stopped training at epoch %d" % stopped_epoch)

					ComputationComplexity_metrics_calculation(ComputationalComplexity_metrics_data)
					
					break
			else:
				wait = 0

				if  IQA_metrics_data['epoch_average_psnr'] > best_epoch_validation_average_psnr :
					best_epoch_validation_average_psnr = IQA_metrics_data['epoch_average_psnr']
					save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, config.model_name, key=1)	
				
				if  IQA_metrics_data['epoch_average_ssim'] > best_epoch_validation_average_ssim :
					best_epoch_validation_average_ssim = IQA_metrics_data['epoch_average_ssim']
					save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, config.model_name, key=2)
					print('best_epoch_validation_average_ssim:', best_epoch_validation_average_ssim)	

				if  IQA_metrics_data['epoch_average_mae'] < best_epoch_validation_average_mae :
					best_epoch_validation_average_mae = IQA_metrics_data['epoch_average_mae']
					save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, config.model_name, key=3)	
				
				if  IQA_metrics_data['epoch_average_lpips'] < best_epoch_validation_average_lpips :
					best_epoch_validation_average_lpips = IQA_metrics_data['epoch_average_lpips']
					save_model(epoch, resultPath_ModelParametersResults, DCE_net, optimizer, config.model_name, key=4)
					print('best_epoch_validation_average_lpips:', best_epoch_validation_average_lpips)

		DCE_net.train()

# **Subpart 13: Record the calculated computation complexity metrics to that csv file**
	with open(csv_ValidationResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of ComputationalComplexity_metrics_data dictionary to that csv file.
		writer = csv.DictWriter(csvfile, fieldnames=ComputationalComplexity_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
		writer.writeheader() # The writer writes the header on the csv file
		writer.writerow(ComputationalComplexity_metrics_data)  # The writer writes the data (value) of ComputationalComplexity_metrics_data dictionary in sequence as a row on the csv file


	
        		


	# -------------------------------------------------------------------
    # train finish
	print("-----Operations completed-----")

	generate_save_TrainingResults_History() # generate and save the training results history as images
	generate_save_ValidationResults_History() # generate and save the validation results history as images



# Task: Make a copy of this script, then->
# 1) [Done] At each epoch, try use reduction = sum to directly add psnr_subtotal of each batch, then at last batch only divide the total psnr_subtotal of all batches to get the average psnr of that epoch. Then verify the results by checking the psnr of each image, by using reduction = 'none'. More info: https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/utilities/distributed.py#L22
# 2) [Done] Verify all IQA metrics and Computational Complexity metrics calculation
# 3) [Done] Add csv to record the average total loss at each epoch, then plot its graph over epoch and save on device. Same goes to IQA metrics.
# 4) Summary: Can take this script as the template to perform model training, but memery management might not efficient for large dataset. Because all training and validation results used for plotting purposes are appended into lists respectively before perform plotting. Instead, use ZTWV_self_lowlight_train_withMetrics_ChangeTorchNoGrad_withsnapshot_scheduler_earlystopping_dataframe.py. It might be more memory efficient because all training and validation results used for plotting purposes are saved into respective csv files first, then convert them into dataframes, then convert into lists respectively before perform plotting. Since much less append() operations involved, its memory efficiency might be better.






