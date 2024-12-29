# This script aims to take images from a specified folder, then generate their corresponding low-light images and high-light images in separate folders respectively, with original filename preserved.
# This script should be used in conjuntion with "dataloader_AdjustBrightness.py"
# Helpful URL:
# 1) https://pytorch.org/vision/master/generated/torchvision.transforms.functional.adjust_brightness.html
# 2) https://www.tutorialspoint.com/how-to-adjust-the-brightness-of-an-image-in-pytorch
# 3) https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
# 4) https://stackoverflow.com/questions/7336096/python-glob-without-the-whole-path-only-the-filename#:~:text=Use%20glob.glob%20%28%22%2A.filetype%22%29%20to%20get%20a%20list%20of,complete%20path%20of%20each%20file%20ending%20with%20.pkl
# 5) https://stackoverflow.com/questions/50878650/pytorch-custom-dataset-dataloader-returns-strings-of-keys-not-tensors

import os
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from dataloader_AdjustBrightness import image_loader 
import argparse
from tqdm import tqdm


def main():
    # image_dataset = image_loader(config.images_path, config.img_size, config.low_light_brightness_factor, config.high_light_brightness_factor)	# set the absolute path of the training dataset	
    low_light_result_path = config.images_src_path.replace('normal-light','low-light') 
    os.makedirs(low_light_result_path, exist_ok=True) # Create the folder to store the low-light images
    high_light_result_path = config.images_src_path.replace('normal-light','high-light')
    os.makedirs(high_light_result_path, exist_ok=True) # Create the folder to store the high-light images

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the GPU as device if it is available, else CPU will be set as the device

    image_dataset = image_loader(config.images_src_path, config.img_size) # Specify the source image dataset folder path to create the custom datset, to be used by the dataloader later	
    image_batch = torch.utils.data.DataLoader(image_dataset, batch_size=config.image_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) # Split the custom dataset into batches of images

    image_batch_bar = tqdm(image_batch, desc="Creating a low-light image and high-light image for each input image in a batch") # To enable the tqdm progress bar feature
    # for batch_idx, batch in enumerate(image_batch_bar):
    #     # print(batch)
    #     print(batch["image_path"])
    #     print(batch["image"])
    
    for batch_idx, (image_paths, images) in enumerate(image_batch_bar): # At each cycle this for block executes, a batch/group/collection of samples will be fetched from image_batch. In other words, iteration stores the batch number (EG: 1st batch, 2nd batch,... provided by enumerate()). In other words, when this for block excutes 1 cycle completely, it means a batch of samples have been processed.
        # print(image_paths) # For each batch, check the absolute path of each image 
        # print(images) # For each batch, check the pixel information of each image 
    
        images = images.to(device) # For each batch, move each image pixel information to the specified device
        low_light_images = F.adjust_brightness(images, config.low_light_brightness_factor) # For each batch, adjust the brightness of each image to be lower to create its corresponding low-light iamge
        high_light_images = F.adjust_brightness(images, config.high_light_brightness_factor) # For each batch, adjust the brightness of each image to be higher to create its corresponding high-light iamge

        for i in range(len(image_paths)):
            torchvision.utils.save_image(low_light_images[i], low_light_result_path + "/" + os.path.basename(image_paths[i])) # Save each low-light image into the low-light folder. save_image will automatically move tensors from GPU to CPU before saving them on disks, because file I/O operations are typically CPU-bound.
            torchvision.utils.save_image(high_light_images[i], high_light_result_path + "/" + os.path.basename(image_paths[i])) # Save each high-light image into the high-light folder
                
  


if __name__ == "__main__": 

    parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.

    # Input Parameters
    parser.add_argument('--images_src_path', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/ImageProcessing_AdjustBrightness/sampling_SICE_Part1_TrainingSet/normal-light") # The source image dataset folder path 
    parser.add_argument('--img_size', type=int, default=416) # The dimensions of the generated images
    parser.add_argument('--low_light_brightness_factor', type=float, default=0.3) # The brightness factor to adjust the image brightness to generate its corresponding low-light image. Range = [0,1]
    parser.add_argument('--high_light_brightness_factor', type=float, default=1.6) # The brightness factor to adjust the image brightness to generate its corresponding high-light image. Range = [1,inf]
    parser.add_argument('--image_batch_size', type=int, default=8) # The number of images to be processed in each batch
    parser.add_argument('--num_workers', type=int, default=4)

    # The parse_args() object (config=self) takes the data (values) you provide to your positional/optional arguments on command line interface or within the () of parse_args(), then converts them into the required data type as mentioned in add_argument() respectively. 
    # So you can access the data of a positional/optional argument by using the syntax args.argument_name (EG: config.lowlight_images_path).
    config = parser.parse_args()

    main()


