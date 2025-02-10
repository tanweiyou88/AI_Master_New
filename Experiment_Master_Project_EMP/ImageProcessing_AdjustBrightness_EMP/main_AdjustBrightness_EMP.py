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

from EMP_Experiment_Master_Project.ImageProcessing_AdjustBrightness_EMP.dataloader_AdjustBrightness_EMP import image_loader 
import argparse
from tqdm import tqdm
import shutil
import glob


def check_lists(list1, list2): # Used to check if 2 lists have the same elements
    list1_sorted = sorted(list1)
    list2_sorted = sorted(list2)

    if list1_sorted == list2_sorted:
        return True
    else:
        return False


def main():

    print("----------------Adjust Brightness Operations Begins----------------")
    # image_dataset = image_loader(config.images_path, config.img_size, config.low_light_brightness_factor, config.high_light_brightness_factor)	# set the absolute path of the training dataset	

    # Create all the required folders
    low_light_result_path = config.images_src_path.replace('normal-light','low-light') 
    low_light_result_train_images_path = os.path.join(low_light_result_path,config.train_folder_name,config.image_folder_name)
    low_light_result_train_labels_path = os.path.join(low_light_result_path,config.train_folder_name,config.label_folder_name)  
    low_light_result_test_images_path = os.path.join(low_light_result_path,config.test_folder_name,config.image_folder_name)
    low_light_result_test_labels_path = os.path.join(low_light_result_path,config.test_folder_name,config.label_folder_name)
    os.makedirs(low_light_result_train_images_path, exist_ok=True) # Create the folder to store the low-light train images
    os.makedirs(low_light_result_train_labels_path, exist_ok=True) # Create the folder to store the low-light train labels
    os.makedirs(low_light_result_test_images_path, exist_ok=True) # Create the folder to store the low-light test images
    os.makedirs(low_light_result_test_labels_path, exist_ok=True) # Create the folder to store the low-light test labels
    high_light_result_path = config.images_src_path.replace('normal-light','high-light')
    high_light_result_train_images_path = os.path.join(high_light_result_path,config.train_folder_name,config.image_folder_name)
    high_light_result_train_labels_path = os.path.join(high_light_result_path,config.train_folder_name,config.label_folder_name)
    high_light_result_test_images_path = os.path.join(high_light_result_path,config.test_folder_name,config.image_folder_name)
    high_light_result_test_labels_path = os.path.join(high_light_result_path,config.test_folder_name,config.label_folder_name)
    os.makedirs(high_light_result_train_images_path, exist_ok=True) # Create the folder to store the high-light train images
    os.makedirs(high_light_result_train_labels_path, exist_ok=True) # Create the folder to store the high-light train labels
    os.makedirs(high_light_result_test_images_path, exist_ok=True) # Create the folder to store the high-light test images
    os.makedirs(high_light_result_test_labels_path, exist_ok=True) # Create the folder to store the high-light test labels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set the GPU as device if it is available, else CPU will be set as the device

    
    folder_list = [config.train_folder_name, config.test_folder_name] 

    # Creating low-light images and high-light images
    print("****Part1: Creating low-light images and high-light images operations begins****")
    for folder in folder_list:
        image_dataset = image_loader(os.path.join(config.images_src_path,folder,config.image_folder_name), config.img_size) # Specify the source image dataset folder path to create the custom datset, to be used by the dataloader later	
        image_batch = torch.utils.data.DataLoader(image_dataset, batch_size=config.image_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) # Split the custom dataset into batches of images

        image_batch_bar = tqdm(image_batch, desc="Creating a low-light image and high-light image for each input image in each batch [{}]".format(folder)) # To enable the tqdm progress bar feature
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
                torchvision.utils.save_image(low_light_images[i], os.path.join(low_light_result_path,folder,config.image_folder_name) + "/" + os.path.basename(image_paths[i])) # Save each low-light image into the low-light folder. save_image will automatically move tensors from GPU to CPU before saving them on disks, because file I/O operations are typically CPU-bound.
                torchvision.utils.save_image(high_light_images[i], os.path.join(high_light_result_path,folder,config.image_folder_name) + "/" + os.path.basename(image_paths[i])) # Save each high-light image into the high-light folder

    print("****Part1: Creating low-light images and high-light images operations completed****")
    print("\n****Part2: Copying and pasting the labels of low-light images and high-light images operations begins****")

    # Copying and pasting the labels of low-light images and high-light images   
    for folder in folder_list: 
        label_filelist = glob.glob(os.path.join(config.images_src_path,folder,config.label_folder_name) + "/*") # Use sequence unpacking method to get the only element from the list. Ref: https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python
        print("Total labels from source dataset folder path [{}]:".format(folder), len(label_filelist))
        for label_file in tqdm(label_filelist, desc="Copying and pasting the labels of low-light images and high-light images [{}]".format(folder)):
            shutil.copy2(label_file, os.path.join(low_light_result_path,folder,config.label_folder_name))
            shutil.copy2(label_file, os.path.join(high_light_result_path,folder,config.label_folder_name))
    
    print("****Part2: Copying and pasting the labels of low-light images and high-light images operations completed****")

    # Get the list of filenames available in the image folder and label folder of all normal-light, low-light, and high-light conditions
    normal_light_train_image_list = os.listdir(os.path.join(config.images_src_path,config.train_folder_name,config.image_folder_name))
    normal_light_test_image_list = os.listdir(os.path.join(config.images_src_path,config.test_folder_name,config.image_folder_name))
    normal_light_train_label_list = os.listdir(os.path.join(config.images_src_path,config.train_folder_name,config.label_folder_name))
    normal_light_test_label_list = os.listdir(os.path.join(config.images_src_path,config.test_folder_name,config.label_folder_name))
    
    low_light_train_image_list = os.listdir(low_light_result_train_images_path)
    low_light_test_image_list = os.listdir(low_light_result_test_images_path)
    low_light_train_label_list = os.listdir(low_light_result_train_labels_path)
    low_light_test_label_list = os.listdir(low_light_result_test_labels_path)

    high_light_train_image_list = os.listdir(high_light_result_train_images_path)
    high_light_test_image_list = os.listdir(high_light_result_test_images_path)
    high_light_train_label_list = os.listdir(high_light_result_train_labels_path)
    high_light_test_label_list = os.listdir(high_light_result_test_labels_path)

    print("\n****Part3: Summary of files checking among the image folder and label folder of all normal-light, low-light, and high-light conditions****")
    # Check train folder
    if check_lists(normal_light_train_image_list, low_light_train_image_list): # check if the 2 folders have the same elements
        # print("The train image folder of normal-light and low-light conditions have the same elements")
        if check_lists(low_light_train_image_list, high_light_train_image_list):
            print("The train image folder of all normal-light, low-light, and high-light conditions have the same elements")
    else: 
        print("The train image folder of normal-light and low-light conditions don't have the same elements")

    if check_lists(normal_light_train_label_list, low_light_train_label_list): # check if the 2 folders have the same elements
        # print("The train label folder of normal-light and low-light conditions have the same elements")
        if check_lists(low_light_train_label_list, high_light_train_label_list):
            print("The train label folder of all normal-light, low-light, and high-light conditions have the same elements")
    else: 
        print("The train label folder of normal-light and low-light conditions don't have the same elements")

    # Check test folder
    if check_lists(normal_light_test_image_list, low_light_test_image_list): # check if the 2 folders have the same elements
        # print("The test image folder of normal-light and low-light conditions have the same elements")
        if check_lists(low_light_test_image_list, high_light_test_image_list):
            print("The test image folder of all normal-light, low-light, and high-light conditions have the same elements")
    else: 
        print("The test image folder of normal-light and low-light conditions don't have the same elements")

    if check_lists(normal_light_test_label_list, low_light_test_label_list): # check if the 2 folders have the same elements
        # print("The test label folder of normal-light and low-light conditions have the same elements")
        if check_lists(low_light_test_label_list, high_light_test_label_list):
            print("The test label folder of all normal-light, low-light, and high-light conditions have the same elements")
    else: 
        print("The test label folder of normal-light and low-light conditions don't have the same elements")

    
    print("----------------Adjust Brightness Operations Completed----------------")


if __name__ == "__main__": 

    parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
    # Input Parameters
    parser.add_argument('--images_src_path', type=str, default="D:/AI_Master_New/PlantDoc.v1-resize-416x416.yolov8-normal-light") # The source image dataset folder path 
    parser.add_argument('--train_folder_name', type=str, default="train") # The folder name that stores images and labels of train datasets 
    parser.add_argument('--test_folder_name', type=str, default="test") # The folder name that stores images and labels of test datasets 
    parser.add_argument('--image_folder_name', type=str, default="images") # The folder name that stores images of train and test datasets 
    parser.add_argument('--label_folder_name', type=str, default="labels") # The folder name that stores labels of train and test datasets
    parser.add_argument('--img_size', type=int, default=416) # The dimensions of the generated images
    parser.add_argument('--low_light_brightness_factor', type=float, default=0.15) # The brightness factor to adjust the image brightness to generate its corresponding low-light image. Range = [0,1]
    parser.add_argument('--high_light_brightness_factor', type=float, default=2.3) # The brightness factor to adjust the image brightness to generate its corresponding high-light image. Range = [1,inf]
    parser.add_argument('--image_batch_size', type=int, default=16) # The number of images to be processed in each batch
    parser.add_argument('--num_workers', type=int, default=4)

    # The parse_args() object (config=self) takes the data (values) you provide to your positional/optional arguments on command line interface or within the () of parse_args(), then converts them into the required data type as mentioned in add_argument() respectively. 
    # So you can access the data of a positional/optional argument by using the syntax args.argument_name (EG: config.lowlight_images_path).
    config = parser.parse_args()

    main()


