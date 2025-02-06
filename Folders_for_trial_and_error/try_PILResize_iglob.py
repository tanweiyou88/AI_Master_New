## Option 1: Resize an image and then save it in a folder
# import os
# import sys
# import torch
# import torch.utils.data as data
# import numpy as np
# from PIL import Image
# import glob
# import random
# import cv2

# img_path = "D:/AI_Master_New/Dataset/LLIE_dataset/SICE_Dataset_Part1/Label/3.JPG" # absolute path of the input image

# image_square_size = 512 # the size of the input image to be resized
# ori_image = Image.open(img_path) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
		
# resized_image = ori_image.resize((image_square_size,image_square_size), Image.LANCZOS) # resize the input image

# folder_path = "D:/AI_Master_New/Folders_for_trial_and_error/ZTWV_ZeroDCE_train_with_val/try_PILResize_ImagePair"
# folder_path_original_images = os.path.join(folder_path, "before") # the folder save original images
# folder_path_resized_images = os.path.join(folder_path, "after") # the folder save resized images

# os.makedirs(folder_path_original_images, exist_ok=True)
# os.makedirs(folder_path_resized_images, exist_ok=True)

# ori_image.save(os.path.join(folder_path_original_images, os.path.basename(img_path))) # save resized images
# resized_image.save(os.path.join(folder_path_resized_images, os.path.basename(img_path))) # save resized images

## Option 2: Resize the images in the original folder and then save them in a new folder
import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
import time
from tqdm import tqdm

try_img_folder_path = "D:/AI_Master_New/Folders_for_trial_and_error/Verify_ResizeSpeed/try_images" # absolute path of the input image
image_square_size = 400 # the size of the input image to be resized

# file_list = glob.iglob(try_img_folder_path+"/*")

start = time.time() # Get the starting time of image enhancement process on a batch of samples, in the unit of second.

for file in tqdm(glob.iglob(try_img_folder_path+"/*")): # For each image in the file list (available inside the current folder)
    
    ori_image = Image.open(file) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
    resized_image = ori_image.resize((image_square_size,image_square_size), Image.LANCZOS) # resize the input image

    # resized_image.save(os.path.join(big_img_folder_path, os.path.basename(file))) # save resized images to a new folder, while remain its filename unchanged
    resized_image.save(os.path.join(try_img_folder_path, os.path.basename(file))) # save resized images to the existing folder, while remain its filename unchanged


duration = (time.time() - start) # Get the duration of image enhancement process on the current batch of samples
print("Duration of processing images:", duration)