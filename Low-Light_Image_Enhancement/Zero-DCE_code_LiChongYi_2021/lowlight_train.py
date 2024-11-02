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
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	DCE_net = model.enhance_net_nopool().cuda() # Move the model to cuda so that the samples in each batch can be processed simultaneously

	DCE_net.apply(weights_init)
	if config.load_pretrain == True: # When NOT set to train mode 
	    DCE_net.load_state_dict(torch.load(config.pretrain_dir)) # load the parameters (weights & biases) of the model obtained/learned at the snapshot (EG: at a particular epoch)
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)	# set the absolute path of the training dataset	
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True) # Split the samples in the training dataset into batches/groups/collections of train_batch_size samples. By default, train_batch_size=8, that's means each batch has 8 samples.



	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()


	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay) # Set the hyperparameters for the optimizer to adjust the model parameters (weights and biases), after each loss.backward() [perform backpropagation = calculate the partial derivative of loss with respect to each weight] of a batch of samples
	
	DCE_net.train() # tells your model that you are training the model. This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation. For instance, in training mode, BatchNorm updates a moving average on each new batch; whereas, for evaluation mode, these updates are frozen. It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader): # At each cycle this for block executes, a batch/group/collection of samples = train_batch_size samples will be fetched from train_loader. In other words, iteration stores the batch number (EG: 1st batch, 2nd batch,... provided by enumerate()) & img_lowlight stores train_batch_size samples of that corresponding batch. In other words, when this for block excutes 1 cycle completely, it means a batch of samples under a particular epoch have been processed.

			img_lowlight = img_lowlight.cuda() # Since this batch of train_batch_size samples are moved to cuda for processinng, they will be processed simultaneously.

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight) # Call the model to enhance this batch of samples, by returning each image enhanced at 4th enhancing iteration, each image enhanced at last enhancing iteration, and the 24 curve parameter maps used to enhance this batch of samples at different stages respectively.
			
			Loss_TV = 200*L_TV(A) # Calculate the illumination smoothness loss of this batch of samples
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight)) # Calculate the spatial consistency loss of this batch of samples

			loss_col = 5*torch.mean(L_color(enhanced_image)) # Calculate the color constancy loss of this batch of samples

			loss_exp = 10*torch.mean(L_exp(enhanced_image)) # Calculate the exposure control loss of this batch of samples
			
			
			# Total loss of this batch of samples (best_loss, considering the weightage of each loss contributed to the total loss)
			loss =  Loss_TV + loss_spa + loss_col + loss_exp 
			#

			
			optimizer.zero_grad() # Remove all previously calculated partial derivatives
			loss.backward() # Perform backpropagation
			torch.nn.utils.clip_grad_norm_(DCE_net.parameters(),config.grad_clip_norm) # Perform Gradient Clipping by Norm to prevent the gradients from becoming excessively large during the training of neural networks, which will lead to exploding gradients problem.
			optimizer.step() # Update the parameters (weights and biases) of the model 

			if ((iteration+1) % config.display_iter) == 0: # at every (display_iter)th iteration
				print("Loss at iteration", iteration+1, ":", loss.item()) # loss.item() means get the value of the 1D tensor called loss. In other words, it gets the total loss value of this batch of samples.
			if ((iteration+1) % config.snapshot_iter) == 0: # at every (snapshot_iter)th iteration
				
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') # Save the parameters (weigths and biases) of the model at the specified snapshot (EG:epoch) as a pth file	



# When this file is run as a script (not get imported to other script), the expression __name__ == "__main__" returns True. 
# The code block under if then runs.
if __name__ == "__main__": 

	parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="D:/AI_Master_New/Low-Light_Image_Enhancemnet/Zero-DCE_code_LiChongYi_2021/data/train_data/") # Add an argument type (optional argument) named lowlight_images_path. The value given to this argument type must be string data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--lr', type=float, default=0.0001) # Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4) # Original batch size
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="D:/AI_Master_New/Low-Light_Image_Enhancemnet/Zero-DCE_code_LiChongYi_2021/snapshots/self_train_snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	# The parse_args() object (config=self) takes the data (values) you provide to your positional/optional arguments on command line interface or within the () of parse_args(), then converts them into the required data type as mentioned in add_argument() respectively. 
	# So you can access the data of a positional/optional argument by using the syntax args.argument_name (EG: config.lowlight_images_path).
	config = parser.parse_args() 

	if not os.path.exists(config.snapshots_folder): # If path=value stored in the config.snapshots_folder (argument type called snapshots_folder stored in config) is not found on the system/computer (means the folder called snapshots has not been created on the system/computer)
		os.mkdir(config.snapshots_folder) # Then automatically create the folder called snapshots on the system/computer, by using the path stored in the config.snapshots_folder


	train(config) # Call the train() and passing all the argument values stored in config to train()








	
