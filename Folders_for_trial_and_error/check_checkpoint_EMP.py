import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import time
import dataloader_WithPair_EMP
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
import pandas as pd
import numpy as np
import random


# function used to initialize the parameters (weights and biases) of the network. This function will be used only when the network is not using the pretrained parameters
def weights_init(m): # A custom function that checks the layer type and applies an appropriate initialization strategy for each case, so that it ensures that the initialization is applied consistently across all layers. It iterating through model layers and systematically apply weight initialization across all layers in a model using the model.apply method. Means in this case, all layers whose name containing "Conv" and "BatchNorm" as part will be initialized using the defined methods.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # if layer whose part of its name having 'Conv' is not found by find(), it will return '-1'
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: # if layer whose part of its name having 'BatchNorm' is not found by find(), it will return '-1'
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




# **Part 1: ArgumentParser, usable only if you run this script**
if __name__ == '__main__':

	parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
	# Input Parameters
	parser.add_argument('--model_name', type=str, default= "ZeroDCE_ori") # Max 30 characters. The random name you give to your created model, so that it appears as the model name at the end of the printed results. This is not involved in the calflop operations, so this information is optional.
	parser.add_argument('--dataset_name', type=str, default= "SICE_Part1") # Max 15 characters
	parser.add_argument('--train_GroundTruth_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/train/GroundTruth") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--train_Input_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/train/Input") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--val_GroundTruth_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/val/GroundTruth") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--val_Input_root', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part1_EMP/val/Input") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--dir_store_results', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/ZeroDCE_ori_EMP") 
	# parser.add_argument('--pretrain_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/snapshots/Epoch99.pth") # The pretrained model parameters provided by the author
	parser.add_argument('--checkpoint_dir', type=str, default= "C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/ZeroDCE_ori_EMP/2025_02_13-17_52_32-SICE_Part1-TrainResultsPool/ModelParam/2025_02_13-17_52_32-ZeroDCE_ori-SICE_Part1-LastEpoch_checkpoint.pt") # The pretrained model parameters obtained by myself
	parser.add_argument('--snapshot_iter', type=int, default=5) # The interval to save the checkpoint
	parser.add_argument('--BatchesNum_ImageGroups_VisualizationSample', type=int, default=2) # At each epoch, save the maximum first N batches of LL_image (Input), enhanced_image (Output), and ori_image (GroundTruth) image groups in validation as the samples to visualize the network performance 
	
	# Training hyperparameters
	parser.add_argument('--load_pretrain', type=bool, default= True) # Set if want to load the pretrained model's parameter. "False" means does not load the pretrained model's parameters; "True" means load the pretrained model's parameters
	parser.add_argument('--image_square_size', type=int, default=256) # The size of the input image to be resized
	parser.add_argument('--num_epochs', type=int, default=6)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=0.0001) # Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--scheduler_step_size', type=int, default = 2) # The step size of scheduler: Period of learning rate decay
	parser.add_argument('--scheduler_gamma', type=float, default = 1) # Multiplicative factor of learning rate decay. "1" means disable scheduler; else, means enable scheduler with that gamma
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



# **Subpart 1: Select the device for computations**
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	random.seed(1) # For deterministic/reproducible results:
	np.random.seed(1) # For deterministic/reproducible results:
	torch.manual_seed(1) # For deterministic/reproducible results: This function ensures the reproducibility of the parameters (weights and biases) of the network which are initialized using random normal functions. The PyTorch RNG is seeded/initialized everytime before the dataloader object is called (at the beginning of each epoch)
	torch.cuda.manual_seed_all(1) # For deterministic/reproducible results:
	torch.backends.cudnn.deterministic=True # For deterministic/reproducible results: Ensures that cuDNN uses deterministic algorithms for convolution operations
	torch.backends.cudnn.benchmark=False # For deterministic/reproducible results: Ensures that CUDA selects the same algorithm each time an application is run

# **Subpart 3: Initialize the network**
	DCE_net = model_ZeroDCE_ori_EMP.enhance_net_nopool().cuda() # Move the model to the specified device so that the samples in each batch can be processed simultaneously

	DCE_net.apply(weights_init)
	

# # **Subpart 5: Initialize the optimizer used for training**
	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay) # Set the hyperparameters for the optimizer to adjust the model parameters (weights and biases), after each loss.backward() [perform backpropagation = calculate the partial derivative of loss with respect to each weight] of a batch of samples
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma, last_epoch=-1) # Decays the learning rate of each parameter group by gamma every step_size epochs.

	if config.load_pretrain == True: # When using the pretrained network parameters

		checkpoint = torch.load(config.checkpoint_dir)
		DCE_net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		

		print("checkpoint['model_state_dict']:", checkpoint['model_state_dict'])
		print("checkpoint['optimizer_state_dict']:", checkpoint['optimizer_state_dict'])
		print("checkpoint['scheduler_state_dict']:", checkpoint['scheduler_state_dict'])
		print("checkpoint['epoch']:", checkpoint['epoch'])
			



