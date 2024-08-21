from numpy import zeros
from numpy import ones
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
from keras.layers import Dropout
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model

# Since pix2pix is a conditional GAN, it takes 2 inputs - image and corresponding label (target image)
# For pix2pix, the label will be another image, called target image. 

# Define the standalone discriminator model
# Given an image sample (each target/generated(fake) image concatenated with its corresponding input image) at the Discriminator's input layer, the Discriminator outputs the likelihood of the image sample being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.
# Think of discriminator as a binary classifier that is classifying image samples at its input layer as real/fake.

# From the paper C64-C128-C256-C512 (C[value] refers to convolution layer, value refers to the number of filters available in that convolution layer)
# At the last (output) layer, perform convolution to downsample the input vector of that layer into a 1-dimensional output, followed by a Sigmoid function.  

def define_discriminator(image_shape):
    
	# Weight initialization
	init = RandomNormal(stddev=0.02) # As described in the original paper
    
	# Input 1: Input image (also called Source image) (The image we want to convert to another image)
	in_src_image = Input(shape=image_shape) # This line initializes the variable in_src_image as the input layer of the model. Each sample provided to this input layer must have size of image_shape, but the batch size of this input layer is omitted (means you can supply any batch size of samples as you want to this input layer as the input, but each sample must have shape of image_shape).
	# Input 2: Target image (The image used to specify the image type we want the generator to generate after training)
	in_target_image = Input(shape=image_shape)  # This line initializes the variable in_target_image as the input layer of the model. Each sample provided to this input layer must have size of image_shape, but the batch size of this input layer is omitted (means you can supply any batch size of samples as you want to this input layer as the input, but each sample must have shape of image_shape).
    
	# Concatenate (channel-wise) each target image with its corresponding input image as a real sample (which will be supplied to the input layer of the discriminator) 
	merged = Concatenate()([in_src_image, in_target_image])
    
	# Here, the discriminator network receives each real sample (input image + target image) to classify if each of the received sample is a real sample.
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged) # Means each input of this convolution layer (merged) is a real sample of  has shape of (32,32,4), each output sample of this convolution layer is (d)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2 
    # Not in the original paper. Comment this block if you want.
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# The output layer provides image patches (means the discriminator network will split an image into image patches, then identify each image patch if it is a real image patch)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# The real sample patch probability score for each input image patch
	patch_out = Activation('sigmoid')(d)
	# Define model
	model = Model([in_src_image, in_target_image], patch_out) # Here we just define like the model created using Model() has 2 input layers (in_src_image layer & in_target_image layer) and 1 output layer (patch_out layer). We omit the batch size here (because the batch size argument is omitted when we define both input layers). However, we can provide a batch size of samples to both of the input layers, and the same batch size of samples will be generated at the output layer, while the dimension of each sample in the batch at different layers follows the ones we set during developing the layers of the model.
	# Compile model
    # The model is trained with a batch size of one image and Adam opt, with a small learning rate and 0.5 beta (The loss for the discriminator is weighted by 50% for each model update). 
    
	opt = adam_v2.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5]) 
	return model

# disc_model = define_discriminator((256,256,3))
# plot_model(disc_model, to_file='disc_model.png', show_shapes=True)

# Now define the generator - in our case we will define a U-net (consists of an encoder and a decoder) as the generator
# Define an encoder block to be used in generator
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# Weight initialization
	init = RandomNormal(stddev=0.02)
	# Add downsampling layer (convolution layer)
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# Conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# Leaky relu activation function
	g = LeakyReLU(alpha=0.2)(g)
	return g

# Define a decoder block to be used in generator
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# Weight initialization
	init = RandomNormal(stddev=0.02)
	# Add upsampling layer (deconvolution layer)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# Add batch normalization
	g = BatchNormalization()(g, training=True)
	# Conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# Merge with skip connection
	g = Concatenate()([g, skip_in]) # This skip connection makes the generator as a unit (like a module), not just a regular encoder decoder.
	# Relu activation function
	g = Activation('relu')(g)
	return g

# Define the standalone generator model - U-net
def define_generator(image_shape=(256,256,3)):
	# Weight initialization
	init = RandomNormal(stddev=0.02)
	# Input image
	in_image = Input(shape=image_shape) # This line initializes the variable in_image as the input layer of the model. Each sample provided to this input layer must have size of image_shape, but the batch size of this input layer is omitted (means you can supply any batch size of samples as you want to this input layer as the input, but each sample must have shape of image_shape).
	# Encoder model: C64(First layer)-C128-C256-C512-C512-C512-C512-C512(Last layer)
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# Bottleneck (The base of the U-Net. This variable stores the reduced dimension of the given input image, similar to the one of autoencoder), no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7) # Means each input of this convolution layer (e7) is a feature map of the input image, each output sample of this convolution layer (d) is a smaller feature map of the same input image
	b = Activation('relu')(b)
	# Decoder model: CD512(First layer)-CD512-CD512-C512-C256-C128-C64(Last layer)
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# Output layer
	g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7) #Modified 
	out_image = Activation('tanh')(g)  # Generates each image whose pixel values are in the range from -1 to 1 (because of Tanh activation function). So also change (preprocess/scale) the pixel values of each input image and target image into the range from -1 to 1.
	# define model
	model = Model(in_image, out_image) # Here we just define like the model created using Model() has 1 input layer (in_image layer) and 1 output layer (out_image layer). We omit the batch size here (because the batch size argument is omitted when we define both input layers). However, we can provide a batch size of samples to both of the input layers, and the same batch size of samples will be generated at the output layer, while the dimension of each sample in the batch at different layers follows the ones we set during developing the layers of the model.
	return model

# gen_model = define_generator((256,256,3))
# plot_model(gen_model, to_file='gen_model.png', show_shapes=True)

# Generator network is trained via GAN combined model. 
# Define the GAN model (combined generator and discriminator model), for updating the weights and biases of the generator network
# Discriminator network is trained separately so here only generator network will be trained by keeping the weights and biases of the discriminator network constant. 
# 
# Important information when defining the layers (structure) of a model:
# 1) We only consider the shape of a sample provided at the input layer, before and after a layer, and generated at the output layer but omit the batch size (if the batch size argument is omitted when we define the input layers). 
# 2) However, when deploying the model, we can provide a batch size of samples to the input layer, and the same batch size of samples will be generated at the output layer. While the dimension of each sample in the batch at different layers follows the ones we set during developing the layers of the model.
#
def define_gan(g_model, d_model, image_shape):
	# Make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False       # Descriminator layers set to untrainable in the GAN model but standalone descriminator will be trainable.
            
	# Input image (also called Source image) (The image we want to convert to another image)
	in_src = Input(shape=image_shape)  # This line initializes the variable in_src as the input layer of the model. Each sample provided to this input layer must have size of image_shape, but the batch size of this input layer is omitted (means you can supply any batch size of samples as you want to this input layer as the input, but each sample must have shape of image_shape).
	# Suppy each input (source) image as input to the generator 
	gen_out = g_model(in_src) # Define the created generator network as a layer of the model to provide each fake sample (each generated(fake) image concatenated with its corresponding input image). Means by omitting the sample batch size (means you can provide any batch size of samples you want), the generator network will provide each fake sample of shape image_shape that is stored in the variable gen_out.
	# Supply each input image and generated(fake) image as inputs to the discriminator
	dis_out = d_model([in_src, gen_out]) # Define the created discriminator network as a layer of the model to provide real image sample probability score. Means by omitting the sample batch size (means you can provide any batch size of samples you want), the discriminator network will receive each input image sample at the input layer called in_src & receive each generated (fake) image sample at the input layer called gen_out. Then, the discriminator network generates each output sample of shape (1,1) [which is the real sample classification probability score].
	# src image as input, generated image and disc. output as outputs
	model = Model(in_src, [dis_out, gen_out]) # Here we just define like the model created using Model() has 1 input layer (in_src layer) and 2 output layers (dis_out layer & gen_out layer). We omit the batch size here (because the batch size argument is omitted when we define both input layers). However, when we want to deploy the model, we can provide any batch size of samples to both of the input layers, and the same batch size of samples will be generated at the output layer, while the dimension of each sample in the batch at different layers follows the ones we set during developing the layers of the model. When we want to deploy the model to make prediction, we just replace the name of the input layer with the variable names that store the training data (features). When we want to deploy the model for training, we just replace the name of the input layer with the variable names that store the training data (features) + the variable name that stores the ground truths.
	# Compile the GAN model
	opt = adam_v2.Adam(lr=0.0002, beta_1=0.5)
    
    #Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE). Authors suggested weighting BCE vs L1 as 1:100.
	model.compile(loss=['binary_crossentropy', 'mae'], 
               optimizer=opt, loss_weights=[1,100])
	
	return model

# Select a batch of random input and target images (samples), returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# Unpack dataset (contains input images and their corresponding target images)
	trainA, trainB = dataset # The variable trainA stores a batch of input images, the variable trainB stores the same batch size of the corresponding target image
	# Choose/select random instances
	ix = randint(0, trainA.shape[0], n_samples) # Select n_samples numbers of input images, by randomly generating n_samples numbers of integers between the range from 0 (low, inclusive) to trainA.shape[0] (high, exclusive). Each randomly generated interger by the function represents the index of the input image features in the variable trainA and the index of the corresponding target image features in the variable trainB. The randomly generated indices are stored in the variable ix. Since each integer is randomly generated between the given range, it is possible to have multiple same integer (index of an input image/sample and index of its corresponding target image/sample) in ix, so that it is possible to use multiple same input and target image sets for training.
	# Load (retrieve) the selected input and target images
	X1, X2 = trainA[ix], trainB[ix] # The variable X1 stores the selected input image features, the variable X2 stores the selected target image features
	# Generate 'real sample' class labels (ground truths of 1). Label=1 indicating they are real
	y = ones((n_samples, patch_shape, patch_shape, 1)) # Here, for each image, all of its patches (with dimension of patch_shape x patch_shape pixels) are assiged with the ground truth value (so the variable y is a 3D array). This is because the discriminator we used here is a PatchGAN, which identifies if each image patch of a specific size is real (instead of identifying if the whole image itself [as a patch] is real in one-shot) 
	return [X1, X2], y # Returns a group of randomly selected input image features, their corresponding target image features, and their corresponding class labels (values of 1)

# Generate a batch of fake images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# Generate each fake instance(image) using each input image as the input
	X = g_model.predict(samples) # The variable samples contains the input image. Deploy the generator model to take each input image and generate each fake image. Each generated fake image is then stored in the variable X.
	# Generate 'fake sample' class labels (ground truth of 0)
	y = zeros((len(X), patch_shape, patch_shape, 1)) # Here, for each image, all of its patches (with dimension of patch_shape x patch_shape pixels) are assiged with the ground truth value (so the variable y is a 3D array). This is because the discriminator we used here is a PatchGAN, which identifies if each image patch of a specific size is real (instead of identifying if the whole image itself [as a patch] is real in one-shot) 
	return X, y # Returns a group of generated images and their corresponding class labels (values of 0)

# This function is just for the purpose of periodically come back and plot how the images generated by the generator network are looking while you're training the GAN model.
# Generator generates samples and save the samples as a plot, and also save the generator model.
# GAN models do not converge, we just want to find a good balance between the generator and the discriminator. Therefore, it makes sense to periodically save the generator model and check how good the generated image looks. 
def summarize_performance(step, g_model, dataset, n_samples=3): # g_model refers to the created generator network, dataset contains a batch of randomly selected input images and their corresponding target images
	# Load a batch/group (size of n_samples) of randomly selected input image features and their corresponding target image features from the variable dataset [ignore their corresponding class labels (values of 1)]
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1) # The variable X_realA stores the selected input image features, he variable X_realB stores the selected target image features. "1" refers to each dimension of the image patch.
	# Generate a batch of generated(fake) images [ignore their corresponding class labels (values of 0)]
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1) # The variable X_fakeB stores the generated (fake) images. "1" refers to each dimension of the image patch.
	# Scale all pixel values from [-1,1] to [0,1] (so we can plot the images to visualize them)
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# Plot real source images (also known as input images, the image we want to convert to another image)
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])
	# Plot generated target images (also known as fake images, generated by the generator network)
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])
	# Plot real target image (also known as target images, the image used to specify the image type we want the generator to generate after training)
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])
	# Save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	# Save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
	
    # Train pix2pix (conditional GAN) models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1): # d_model refers to the created discriminator network, g_model refers to the created generator network, gan_model refers to the created GAN model, dataset contains a batch of randomly selected input images and their corresponding target images, n_epochs refers to the total number of epochs for GAN model training (here, default[if no input for this argument] is 100), n_batch refers to the number of samples in a batch of samples (also means the batch size) (here, default[if no input for this argument] is 1, means only 1 sample out of all the training samples is prossed in one go). 
	# Determine each image patch shape output by the discriminator network
	n_patch = d_model.output_shape[1]
	# Unpack dataset (contains input images and their corresponding target images)
	trainA, trainB = dataset # The variable trainA stores a batch of input images, the variable trainB stores the same batch size of the corresponding target image
	# Calculate the number of batches per training epoch (means calculate need to split all the samples for training into how many batch). n_batch refers to the number of samples for training in each batch (can treat a batch like a small group).
	bat_per_epo = int(len(trainA) / n_batch)
	# Calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs # number of steps = number of epochs
	# Manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# Update discriminator for real samples [a real sample = an input image + its corresponding target image] (here, we update the parameters [weights and biases] of the discriminator network everytime after it has processed a batch of samples [by default, only 1 sample in a batch])
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real) # The discriminator training is performed by taking the randomly selected input images and their corresponding target images as the features, while taking the 'real sample' class labels (ground truths of 1) as the target (ground truth)
		# Update discriminator for generated (fake) samples [a fake sample = an input image + its corresponding fake image] (here, we update the parameters [weights and biases] of the discriminator network everytime after it has processed a batch of samples [by default, only 1 sample in a batch])
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake) # The discriminator training is performed by taking the randomly selected input images and the fake imges randomly generated by the generator network as the features, while taking the 'fake sample' class labels (ground truths of 0) as the target (ground truth)
		# Update the generator (here, we update the parameters [weights and biases] of the generator network everytime after it has processed a batch of samples [by default, only 1 sample in a batch])
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB]) # The generator network training is performed through using the GAN model, by taking the randomly selected input images as the feature, while taking the 'real sample' class labels (ground truths of 1) and the fake imges randomly generated by the generator network as the targets (ground truths). the 'real sample' class labels (ground truths of 1) [y_real] is involved in calculating the adversarial loss (BCE);the fake imges randomly generated by the generator network [X_realB] is involved in calculating the L1 loss (MAE).
		# Summarize the GAN model performance by printing the losses at each iteration
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss)) # >step number, d_loss1 [d1] refers to the discriminator loss when training on real samples, d_loss2 [d2] refers to the discriminator loss when training on fake samples, g_loss [g] refers to the generator loss
		# Summarize the generator model performance by showing its generated images periodically
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)
			
