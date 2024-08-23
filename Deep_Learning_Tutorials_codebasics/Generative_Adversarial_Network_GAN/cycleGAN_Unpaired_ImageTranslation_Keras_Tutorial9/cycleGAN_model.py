from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import adam_v2
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

#Download instance norm. code from the link above.
#Or install keras_contrib using guidelines here: https://github.com/keras-team/keras-contrib 
from instancenormalization import InstanceNormalization  

from matplotlib import pyplot

# discriminator model (70x70 patchGAN). The only difference between the discriminator of the cycleGAN here and the one in pix2pix is the discriminator of the cycleGAN here uses the instance normalization.
# C64-C128-C256-C512
#After the last layer, conv to 1-dimensional output, followed by a Sigmoid function.  
# The “axis” argument is set to -1 for instance norm. to ensure that features are normalized per feature map.
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2 
    # Not in the original paper. Comment this block if you want.
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
    #The model is trained with a batch size of one image and Adam opt. 
    #with a small learning rate and 0.5 beta. 
    #The loss for the discriminator is weighted by 50% for each model update.
    #This slows down changes to the discriminator relative to the generator model during training.
	model.compile(loss='mse', optimizer=adam_v2.Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# generator a resnet block to be used in the generator. The resnet block is all about taking the output from the convolution layers to concatenate with the corresponding input of the resnet block.
# residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layers.
def resnet_block(n_filters, input_layer): 
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

# define the  generator model - encoder-decoder type architecture

#c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1. 
#dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
# Rk denotes a residual block that contains two 3 × 3 convolutional layers
# uk denotes a 3 × 3 fractional-strided-Convolution InstanceNorm-ReLU layer with k filters and stride 1/2

#The network with 6 residual blocks consists of:
#c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

#The network with 9 residual blocks consists of:
#c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3

def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define a composite model for updating generators by adversarial and cycle loss
#We define a composite model that will be used to train each generator separately. 
def define_composite_model(g_model_1, d_model, g_model_2, image_shape): # g_model_1 and d_model is a pair (means d_model is the corresponding discriminator of the generator called g_model_1
	# Make the generator of interest trainable as we will be updating these weights.
    #by keeping other models constant.
    #Remember that we use this same function to train both generators,
    #one generator at a time. 
	g_model_1.trainable = True
	# mark the corresponding discriminator of the generator called g_model_1 and second generator as non-trainable
	d_model.trainable = False
	g_model_2.trainable = False
    
	# adversarial loss
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity loss
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# cycle loss - forward
	output_f = g_model_2(gen1_out)
	# cycle loss - backward
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
    
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	
    # define the optimizer
	opt = adam_v2.Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], 
               loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
#Remember that for real images the label (y) is 1. 
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
#Remember that for fake images the label (y) is 0. 
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake images
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# periodically save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
	
# periodically generate images using the save model and plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot n_samples randomly selected real images and its corresponding fake images generated by the selected generator model as a 2x(n_samples) subplot/grid
	# plot real images at the 1st row of the subplot
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image  at the 2nd row of the subplot
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()
	

# update image pool/buffer (that can store max_size images) for fake images to reduce model oscillation
# update discriminators using a history of generated images 
#rather than the ones produced by the latest generators.
#Original paper recommended keeping an image buffer that stores 
#the 50 previously created images.

def update_image_pool(pool, images, max_size=50): # pools refers to a image pool of a particular domain that stores max_size numbers of previously generated fake images of that particular domain; images refers to a variable that stores currently generated fake images of a particular domain (same as the domain of the selected image pool);
	selected = list() # the variable selected will store the selected fake images (each fake image is added to the the variable selected either [directly take the current fake image provided in that iteration] or [take a previously generated fake image from the image pool, before replacing that taken fake image in the image pool with the current fake image provided in that iteration]) that eventually will be used for training. The number of fake images in the variable selected is same as the one of the variable images.
	for image in images:
		if len(pool) < max_size: # when the image pool not yet full (still have room for fake images)
			# stock the pool
			pool.append(image) # add the current fake image to the image pool (the image pool is updated with a current fake image)
			selected.append(image) # add the current fake image to the variable selected
		elif random() < 0.5: # When the image pool is full, and if the randomly generated number is < 0.5 (the probability of 50/50 will execute this command)
			# use image, but don't add it to the image pool
			selected.append(image) # add the current fake image to the variable selected
		else: # When the image pool is full, and if the randomly generated number is >= 0.5 (the probability of 50/50 will execute this command)
			# replace an existing image and use replaced image
			ix = randint(0, len(pool)) # randomly select a previously generated fake image
			selected.append(pool[ix]) # add the selected previously generated fake image to the variable selected
			pool[ix] = image # replace the selected previously generated fake image with the current fake image (the image pool is updated with a current fake image)
	return asarray(selected) # returns the selected fake images that eventually will be used for training


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1): # d_model refers to a discriminator, g_model refers to a generator, c_model refers to a composite mode (GAN model), A refers to the domain/type A, B refers to the domain/type B, AtoB meanns the model input is domain/type A and its output is domain/type B.
	# define properties of the training run
	n_epochs, n_batch, = epochs, 1  #batch size fixed to 1 as suggested in the paper
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset # trainA stores the real domain A images(samples); trainB stores the real domain B images(samples)
	# prepare image pool for fake images
	poolA, poolB = list(), list() # poolA will be used to store max_size numbers of previously generated fake domain A images(samples); poolB will be used to store max_size numbers of previously generated fake domain B images(samples)
	# calculate the number of batches per training epoch (also same as the number of iterations per epoch)
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
    
	# manually enumerate epochs
	for i in range(n_steps):
		# Part 1: select a batch of real samples from each domain (A and B)
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch) # X_realA stores a batch(group) of real domain A images(samples); y_realA stores a batch(group) of "real image class" target(ground truth) labels [values of 1] for each image(sample) in the X_realA.
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch) # X_realB stores a batch(group) of real domain B images(samples); y_realB stores a batch(group) of "real image class" target(ground truth) labels [values of 1] for each image(sample) in the X_realB.
		# Part 2: generate a batch of fake samples using both B to A and A to B generators.
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch) # X_fakeA stores a batch(group) of fake domain A images(samples) generated by the generator called g_model_BtoA; y_fakeA stores a batch(group) of "fake image class" target(ground truth) labels [values of 0] for each image(sample) in the X_fakeA.
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch) # X_fakeB stores a batch(group) of fake domain B images(samples) generated by the generator called g_model_AtoB; y_fakeB stores a batch(group) of "fake image class" target(ground truth) labels [values of 0] for each image(sample) in the X_fakeB.
		# Part 3: update fake images in the pool. Remember that the paper suggstes a buffer of 50 images (max_size=50)
		X_fakeA = update_image_pool(poolA, X_fakeA) # X_fakeA stores a batch(group) of the selected fake domain A images (a combination of currently generated fake domain A images in this epoch and previously generated fake domain A images taken from poolA) that eventually will be used for training
		X_fakeB = update_image_pool(poolB, X_fakeB) # X_fakeB stores a batch(group) of the selected fake domain B images (a combination of currently generated fake domain B images in this epoch and previously generated fake domain B images taken from poolB) that eventually will be used for training
        
		# Part 4: update (means train the model and get its loss) generator B->A (g_model_BtoA) via the composite model (c_model_BtoA)
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA]) # g_loss2 refers to the loss the discriminator A classifies the fake domain A images as the real domain A images (adversarial loss)
		# Train the generator model called g_model_A by calling the composite model called c_model_BtoA, using 2 inputs (variable X_realB and X_realA). 
		# Each sample in the variable X_realB will be provided to the composite model input layer called input_gen, while each sample in the variable X_realA will be provided to the composite model input layer called input_id.
		# Each sample in the variable y_realA is compared with each sample at the composite model output layer called output_d, to calculate the adversarial loss as MSE (which is returned to the variable g_loss2 here). Then, the parameters (weightages and biases) of the generator (g_model_BtoA) is updated. This loss has a weightage of 1.
		# Each sample in the variable X_realA is compared with each sample at the composite model output layer called output_id, to calculate the identity loss as MAE/L1 (which is not returned here, due to the second '_'). This loss has a weightage of 5.
		# Each sample in the variable X_realB is compared with each sample at the composite model output layer called output_f, to calculate the forward cycle loss as MAE/L1 (which is not returned here, due to the third '_'). This loss has a weightage of 10.
		# Each sample in the variable X_realA is compared with each sample at the composite model output layer called output_b, to calculate the backward cycle loss as MAE/L1 (which is not returned here, due to the forth '_'). This loss has a weightage of 10.


		# Part 5: update discriminator for A (d_model_A) -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA) # dA_loss1 refers to the loss the discriminator A classifies the real domain A images as the real domain A images
		# Train the discriminator model called d_model_A by using 1 input (variable X_realA, the real domain A images), then update the parameters (weights and biases) of the discriminator model (d_model_A).
		# This means each sample in the variable X_realA will be provided to the discriminator model input layer called in_image. 
		# Each sample in the variable y_realA is compared with each sample at the discriminator model output layer called patch_out, to calculate the loss as MSE (which is returned to the variable dA_loss1 here). Then, the parameters (weightages and biases) of the discriminator (d_model_A) is updated. This loss has a weightage of 0.5.

		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA) # dA_loss2 refers to the loss the discriminator A classifies the fake domain A images as the real domain A images
		# Train the discriminator model called d_model_A by using 1 input (variable X_fakeA, the fake domain A images), then update the parameters (weights and biases) of the discriminator model (d_model_A).
		# This means each sample in the variable X_fakeA will be provided to the discriminator model input layer called in_image.
		# Each sample in the variable y_fakeA is compared with each sample at the discriminator model output layer called patch_out, to calculate the loss as MSE (which is returned to the variable dA_loss2 here). Then, the parameters (weightages and biases) of the discriminator (d_model_A) is updated. This loss has a weightage of 0.5.

        
		# Part 6: update generator A->B (g_model_AtoB) via the composite model (c_model_AtoB)
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])  # g_loss1 refers to the loss the discriminator B classifies the fake domain B images as the real domain B images (adversarial loss)
		# Train the generator model called g_model_AtoB by calling the composite model called c_model_AtoB, using 2 inputs (variable X_realA and X_realB). 
		# Each sample in the variable X_realA will be provided to the composite model input layer called input_gen, while each sample in the variable X_realB will be provided to the composite model input layer called input_id. 
		# Each sample in the variable y_realB is compared with each sample at the composite model output layer called output_d, to calculate the adversarial loss as MSE (which is returned to the variable g_loss1 here). Then, the parameters (weightages and biases) of the generator (g_model_AtoB) is updated. This loss has a weightage of 1.
		# Each sample in the variable X_realB is compared with each sample at the composite model output layer called output_id, to calculate the identity loss as MAE/L1 (which is not returned here, due to the second '_'). This loss has a weightage of 5.
		# Each sample in the variable X_realA is compared with each sample at the composite model output layer called output_f, to calculate the forward cycle loss as MAE/L1 (which is not returned here, due to the third '_'). This loss has a weightage of 10.
		# Each sample in the variable X_realB is compared with each sample at the composite model output layer called output_b, to calculate the backward cycle loss as MAE/L1 (which is not returned here, due to the forth '_'). This loss has a weightage of 10.


		# Part 7: update discriminator for B (d_model_B) -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB) # dB_loss1 refers to the loss the discriminator B classifies the real domain B images as the real domain B images
		# Train the discriminator model called d_model_B by using 1 input (variable X_realB, the real domain B images), then update the parameters (weights and biases) of the discriminator model (d_model_B).
		# This means each sample in the variable X_realB will be provided to the discriminator model input layer called in_image. 
		# Each sample in the variable y_realB is compared with each sample at the discriminator model output layer called patch_out, to calculate the loss as MSE (which is returned to the variable dB_loss1 here). Then, the parameters (weightages and biases) of the discriminator (d_model_B) is updated. This loss has a weightage of 0.5.

		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB) # dB_loss2 refers to the loss the discriminator B classifies the fake domain B images as the real domain B images
		# Train the discriminator model called d_model_B by using 1 input (variable X_fakeB, the fake domain B images), then update the parameters (weights and biases) of the discriminator model (d_model_B).
		# This means each sample in the variable X_fakeB will be provided to the discriminator model input layer called in_image. 
		# Each sample in the variable y_fakeB is compared with each sample at the discriminator model output layer called patch_out, to calculate the loss as MSE (which is returned to the variable dB_loss2 here). Then, the parameters (weightages and biases) of the discriminator (d_model_B) is updated. This loss has a weightage of 0.5.


        # Part 8: summarize performance
        # Since our batch size =1, the number of iterations would be same as the size of our dataset.
        # In one epoch you'd have iterations equal to the number of images.
        # If you have 100 images then 1 epoch would be 100 iterations
		print('Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
			# g_loss2 refers to the loss the discriminator A classifies the fake domain A images as the real domain A images (adversarial loss)
			# dA_loss1 refers to the loss the discriminator A classifies the real domain A images as the real domain A images
			# dA_loss2 refers to the loss the discriminator A classifies the fake domain A images as the real domain A images
			# g_loss1 refers to the loss the discriminator B classifies the fake domain B images as the real domain B images (adversarial loss)
			# dB_loss1 refers to the loss the discriminator B classifies the real domain B images as the real domain B images
			# dB_loss2 refers to the loss the discriminator B classifies the fake domain B images as the real domain B images
		
		# Evaluate the model performance periodically
        # If batch size (total images)=100, performance will be summarized after every 75th iteration.
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot A->B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB') # Plot the subplot to show the performance of the trained g_model_AtoB (take a real domain A image to generate a fake domain B image)
			# plot B->A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA') # Plot the subplot to show the performance of the trained g_model_AtoB (take a real domain B image to generate a fake domain A image) 
		if (i+1) % (bat_per_epo * 5) == 0:
			# Save all the trained generator models
            # If batch size (total images)=100, model will be saved after every 75th iteration x 5 = 375 iterations.
			save_models(i, g_model_AtoB, g_model_BtoA) # Save all of the trained generator models called g_model_AtoB and g_model_BtoA respectively. So later you can deploy the correct trained generator model to either (taking a real domain A image to generate a fake domain B image) or (taking a real domain B image to generate a fake domain A image). 
