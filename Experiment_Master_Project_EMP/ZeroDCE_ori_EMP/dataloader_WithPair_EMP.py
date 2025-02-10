import os
import torch
import numpy as np
from PIL import Image
from os.path import join
import cv2
import glob
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    # The endswith() method in Python is used to check if a string ends with a specified suffix. It returns True if the string ends with the specified suffix, otherwise it returns False.
    # Python any() function returns True if any of the elements of a given iterable( List, Dictionary, Tuple, set, etc) are True else it returns False.. More info: https://www.geeksforgeeks.org/python-any-function/
    # Equivalent to:
    # for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
    #   result = filename.endswith(extension)
    #   return any(result)


class LLIEDataset(torch.utils.data.Dataset):
    # def __init__(self, ori_root, lowlight_root, transforms, istrain = False, isdemo = False, dataset_type = 'LOL-v1'):
    def __init__(self, GroundTruth_root, Input_root, image_square_size):
        self.Input_root = Input_root
        self.GroundTruth_root = GroundTruth_root
        self.matching_dict = {}
        self.file_list = []
        # self.istrain = istrain
        self.get_image_pair_list() # This function takes all the filepaths available in the specified folder to initialize/create a list as the dataset. So that now file_list is not empty anymore.
        # self.transforms = transforms
        # self.isdemo = isdemo
        self.size = image_square_size # The width and height of each image, in pixel
        print("Total sample numbers:", len(self.file_list)) 

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]

        ori_image = Image.open(ori_image_name).convert('RGB') # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
        ori_image = ori_image.resize((self.size,self.size), Image.LANCZOS) # Resize the opened image to (self.size,self.size) with Image.ANTIALIAS filter for anti-aliasing, using PIL. Then, replace the opened image with its resized image at the given image's absolute path. **ANTIALIAS was removed in Pillow 10.0.0 (after being deprecated through many previous versions). Now you need to use PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.(This is the exact same algorithm that ANTIALIAS referred to, you just can no longer access it through the name ANTIALIAS.)**
        ori_image = (np.asarray(ori_image)/255.0) # Convert the resized image from Image object into Numpy array, then replace the resize image of Image object version with the Numpy array version one. Then, only normalize the pixel values (features) of the resized image to the range [0,1].
        ori_image = torch.from_numpy(ori_image).float() # Convert the pixel values (features) of the resized image from Numpy array into tensor, with the data type of float32.

        ll_image = Image.open(ll_image_name).convert('RGB') # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
        ll_image = ll_image.resize((self.size,self.size), Image.LANCZOS) # Resize the opened image to (self.size,self.size) with Image.ANTIALIAS filter for anti-aliasing, using PIL. Then, replace the opened image with its resized image at the given image's absolute path. **ANTIALIAS was removed in Pillow 10.0.0 (after being deprecated through many previous versions). Now you need to use PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.(This is the exact same algorithm that ANTIALIAS referred to, you just can no longer access it through the name ANTIALIAS.)**
        ll_image = (np.asarray(ll_image)/255.0) # Convert the resized image from Image object into Numpy array, then replace the resize image of Image object version with the Numpy array version one. Then, only normalize the pixel values (features) of the resized image to the range [0,1].
        ll_image = torch.from_numpy(ll_image).float() # Convert the pixel values (features) of the resized image from Numpy array into tensor, with the data type of float32.

        

        return ori_image.permute(2,0,1), ll_image.permute(2,0,1)

    def __len__(self):
        return len(self.file_list)
    
    def get_image_pair_list(self):

        # print(os.listdir(self.Input_root))
        image_name_list = [join(self.Input_root, x) for x in sorted(os.listdir(self.Input_root), key=len) if is_image_file(x)] # Get the list of filenames available in "self.lowlight_root" first. Then for each filename, chech if its string ends with a specific suffix (file extension). If yes, then concatenate the folder absolute path with the filename as a single element and then store in the "image_name_list".
        for key in image_name_list: # For each file absolute path available in the image_name_list
            # print("key:", key) # The file absolute path
            key = key.split("/")[-1] # Get the path in the format of "subfolder + filename.extension"
            # print("key.split:", key)
            if os.name == 'nt': # 'nt' means that you are running windows. More info: https://stackoverflow.com/questions/22321397/python-os-name-return-nt-on-windows-7
                key = key.split("\\")[-1] # Get the "filename.extension". In python, "\\" refers to "\". If you include "\" in python, error occurs. More info: https://stackoverflow.com/questions/70780266/unterminated-string-literal
                # print("key.split, os.name == 'nt':", key)
            # print("Append 1st part:", os.path.join(self.ori_root, key))
            # print("Append 2nd part:", os.path.join(self.lowlight_root, key))
            self.file_list.append([os.path.join(self.GroundTruth_root, key), 
                                os.path.join(self.Input_root, key)])
            # print('file_list:', self.file_list)
                
        # elif dataset_type == 'LOL-v2' or dataset_type == 'LOL-v2-real' or dataset_type == 'LOL-v2-Syn':
        #     if self.istrain:
        #         Real_Low_root = join(self.lowlight_root,'Real_captured', 'Train', "Low")
        #         Synthetic_Low_root = join(self.lowlight_root,'Synthetic', 'Train', "Low")
        #         Real_High_root = join(self.ori_root,'Real_captured', 'Train', "Normal")
        #         Synthetic_High_root = join(self.ori_root,'Synthetic', 'Train', "Normal")
        #     else:
        #         Real_Low_root = join(self.lowlight_root,'Real_captured', 'Test', "Low")
        #         Synthetic_Low_root = join(self.lowlight_root,'Synthetic', 'Test', "Low")
        #         Real_High_root = join(self.ori_root,'Real_captured', 'Test', "Normal")
        #         Synthetic_High_root = join(self.ori_root,'Synthetic', 'Test', "Normal")
            
        #     # For Real
        #     if dataset_type == 'LOL-v2-Syn':
        #         Real_name_list =[]
        #     else:
        #         Real_name_list = [join(Real_Low_root, x) for x in os.listdir(Real_Low_root) if is_image_file(x)]
            
        #     for key in Real_name_list:
        #         key = key.split("/")[-1]
        #         if os.name == 'nt':
        #             key = key.split("\\")[-1]
        #         self.file_list.append([os.path.join(Real_High_root, 'normal'+key[3:]), 
        #                             os.path.join(Real_Low_root, key)])
            
            # For Synthetic

        #     if dataset_type == 'LOL-v2-real':
        #         Synthetic_name_list =[]
        #     else:
        #         Synthetic_name_list = [join(Synthetic_Low_root, x) for x in os.listdir(Synthetic_Low_root) if is_image_file(x)]
            
        #     for key in Synthetic_name_list:
        #         key = key.split("/")[-1]
        #         if os.name == 'nt':
        #             key = key.split("\\")[-1]
        #         self.file_list.append([os.path.join(Synthetic_High_root, key), 
        #                             os.path.join(Synthetic_Low_root, key)])
        
        # elif dataset_type == 'RESIDE':
        #     image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
        #     # if self.istrain:
        #     if os.path.isfile( os.path.join(self.ori_root, image_name_list[0].split('_')[0]+'.jpg')):
        #         FileE = '.jpg'
        #     else:
        #         FileE = '.png'
        #     for key in image_name_list:
        #         key = key.split("/")[-1]
        #         if os.name == 'nt':
        #             key = key.split("\\")[-1]
        #         self.file_list.append([os.path.join(self.ori_root, key.split('_')[0]+FileE), 
        #                             os.path.join(self.lowlight_root,key)])   
        # elif dataset_type == 'expe':
        #     image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
        #     if os.path.isfile( os.path.join(self.ori_root, '_'.join(image_name_list[0].split('_')[:-1])+'.jpg')):
        #         FileE = '.jpg'
        #     else:
        #         FileE = '.png'
        #     for key in image_name_list:
        #         key = key.split("/")[-1]
        #         if os.name == 'nt':
        #             key = key.split("\\")[-1]
        #         self.file_list.append([os.path.join(self.ori_root, '_'.join(key.split('_')[:-1])+FileE), 
        #                             os.path.join(self.lowlight_root,key)])   
        # elif dataset_type == 'VE-LOL':
        #     image_name_list = [join(self.lowlight_root, x) for x in os.listdir(self.lowlight_root) if is_image_file(x)]
        #     for key in image_name_list:
        #         key = key.split("/")[-1]
        #         if os.name == 'nt':
        #             key = key.split("\\")[-1]
        #         self.file_list.append([os.path.join(self.ori_root, key.replace('low', 'normal',)), 
        #                             os.path.join(self.lowlight_root, key)])
        # else:
        #     raise ValueError(str(dataset_type) + "does not support! Please change your dataset type")
                
        # if self.istrain or (dataset_type[:6] == 'LOL-v2'):
        #     random.shuffle(self.file_list)

    # def add_dataset(self, ori_root, lowlight_root, dataset_type = 'LOL-v1',):
    #     self.lowlight_root = lowlight_root
    #     self.ori_root = ori_root
    #     self.get_image_pair_list(dataset_type)

# class LLIE_Dataset(LLIEDataset):
#     def __init__(self, ori_root, lowlight_root, transforms, istrain = True):
#         self.lowlight_root = lowlight_root
#         self.ori_root = ori_root
#         self.image_name_list = glob.glob(os.path.join(self.lowlight_root, '*.png'))
#         self.matching_dict = {}
#         self.file_list = []
#         self.istrain = istrain
#         self.get_image_pair_list()
#         self.transforms = transforms
#         print("Total data examples:", len(self.file_list))

#     def __getitem__(self, item):
#         """
#         :param item:
#         :return: haze_img, ori_img
#         """
#         ori_image_name, ll_image_name = self.file_list[item]
#         ori_image = self.transforms(
#             Image.open(ori_image_name)
#             )

#         LL_image_PIL = Image.open(ll_image_name)
#         LL_image = self.transforms(
#             LL_image_PIL
#             )
        
#         return ori_image, LL_image

#     def __len__(self):
#         return len(self.file_list)