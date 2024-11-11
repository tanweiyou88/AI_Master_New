# Obtain the script from: https://github.com/realnghon/Zero-DCE_Enhanced/blob/master/Zero-DCE_code/PSNR.py
# Each ground truth image and its corresponding enhanced/output image must have the same filename. But they can be located at different folders (different root directory).
# Peak Signal-to-Noise Ratio (PSNR) is defined as the ratio of the maximum possible signal value to the value of the noise that affects the fidelity of its representation.
# PSNR is a general metric for mentioning the quality of an image or any video stream. It aids in determining the degree of information loss or degradation brought on by compression. The apparent quality of the compressed image or video increases as PSNR increases because distortion becomes less noticeable.

# from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchmetrics import PeakSignalNoiseRatio as PSNR
from skimage.io import imread
import numpy as np
import os

def calculate_psnr(dir1, dir2):
    # List comprehension part: For every f=filename in os.listdir(dir1)=(a list of filenames available in dir1), if os.path.isfile(os.path.join(dir1, f)=True=(if a filepath represented as os.path.join(dir1, f) exists), then add f=filename to a new list. After the list comprehension is implemented, only sort the elements/entries in that new list and then add them to the variable called files1.
    files1 = sorted([f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))])
    # List comprehension part: For every f=filename in os.listdir(dir2)=(a list of filenames available in dir2), if os.path.isfile(os.path.join(dir2, f)=True=(if a filepath represented as os.path.join(dir2, f) exists), then add f=filename to a new list. After the list comprehension is implemented, only sort the elements/entries in that new list and then add them to the variable called files2.
    files2 = sorted([f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))])
    psnr_values = [] # create an empty list called psnr_values
    img1_nparray = torch.empty(0, 3, 256, 256) # create an empty tensor that has no element inside, but each of its element should have the shape of (3,256,256)
    img2_nparray = torch.empty(0, 3, 256, 256) # create an empty tensor that has no element inside, but each of its element should have the shape of (3,256,256)
    

    for file in files1: # For each file=filename=(ground truth image) in files1=(the created new list of sorted filenames available in folder dir1). Each iteration will get a groundtruth-enhanced images pair.
        if file in files2: # Check if file=filename is present in files2=(the created new list of sorted filenames available in folder dir2). A True will be returned if it exists.
            # Get a groundtruth-enhanced images pair
            img1 = imread(os.path.join(dir1, file)) # load the ground truth image whose path=os.path.join(dir1, file), then save as variable img1
            img2 = imread(os.path.join(dir2, file)) # load the enhanced/output image whose path=os.path.join(dir2, file), then save as variable img2
            # print(img1.shape)
            # 确保两个图片尺寸一致
            # If want to resize an image, it is easier to downsize an image than it is to enlarge an image. This is because when an image is enlarged, the photo editor must create and add new pixel information -- based on its best guesses (interpolation, where the software estimates the color of the new pixels based on the color of existing pixels. These algorithms, while sophisticated, can't invent details that aren't present in the original image) -- to achieve a larger size which typically results in either a very pixelated or very soft and blurry looking image. However, if an image is downsized, some pixels are required to be removed only, affecting the image quality lesser. However, both upsampling and downsampling image will lead to quality loss.  
            # That's why to ensure both the ground truth image and its corresponding enhanced/output image have the same dimensions/sizes, we use downsampling. So, if the images of the groundtruth-enhanced images pair have different dimensions, we resize both images respectively using the smallest height dimension and smallest width dimension.
            # img1 = np.resize(img1, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            # img2 = np.resize(img2, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            img1 = np.resize(img1, (256, 256, 3))
            # print(img1.shape)
            img1 = (np.asarray(img1)/255.0) 
            img1 = torch.from_numpy(img1).float()  # Convert the pixel values (features) of the image from Numpy array into tensor, with the data type of float32. So now the image becomes a 3D tensor.
            # print(img1.shape)
            img1 = img1.permute(2,0,1) # Use permute() to rearrange the dimensions of the image from [height(H),width(W),channels(C)] to [channels(C),height(H),width(W)]
            img1 = img1.unsqueeze(0)
            # print(img1.shape)
            img1_nparray = torch.cat((img1_nparray,img1), dim=0)


            img2 = np.resize(img2, (256, 256, 3))
            img2 = (np.asarray(img2)/255.0) 
            img2 = torch.from_numpy(img2).float()  # Convert the pixel values (features) of the image from Numpy array into tensor, with the data type of float32. So now the image becomes a 3D tensor.
            img2 = img2.permute(2,0,1) # Use permute() to rearrange the dimensions of the image from [height(H),width(W),channels(C)] to [channels(C),height(H),width(W)]
            img2 = img2.unsqueeze(0)
            img2_nparray = torch.cat((img2_nparray,img2), dim=0)

            # value = psnr(img1, img2) # Calculate the PSNR (a function of scikit-image) on this groundtruth-enhanced images pair. skimage.metrics.peak_signal_noise_ratio only calculates the PSNR of an image pair, in contrast to torchmetrics.PeakSignalNoiseRatio.
            # psnr_values.append(value) # Add this calculated PSNR to the psnr_values list, as a new element/entry

    # Usage of Ternary operator: If pnsr_values is NOT an empty list (at least 1 element/entry is available in the list), then execute np.mean(psnr_values)=([Sum all elements/entries in the list called psnr_values]/[Total number of groundtruth-enhanced images pair, represented as the number of elements/entries in in the list called psnr_values]), then store the result in the variable average_psnr. Else (if pnsr_values is an empty list [no element/entry is available in the list]), store the value 0 in the average_psnr.
    # average_psnr = np.mean(psnr_values) if psnr_values else 0
    # img1_nparray = img1_nparray.cuda().unsqueeze(0)
    # img2_nparray = img2_nparray.cuda().unsqueeze(0)
    img1_nparray = img1_nparray.cuda()
    img2_nparray = img2_nparray.cuda()
    average_psnr = psnr(img1_nparray, img2_nparray).item() # torchmetrics.PeakSignalNoiseRatio first calculates the PSNR of each image pair in a batch of images, then return the average PSNR of all image pairs in that batch. The mathematical concept behinds is: (average PSNR)=([Summation of PSNR of all image pairs in a batch of images]/[Total number of image pairs in that batch of images, which is same as the batch size]) of that batch of images.
    return average_psnr

# Concept/Operating principle of calculating psnr in this script [performed by calculate_psnr()]:
# 1. For a given folder contains ground truth images (represented by ground_truth_folder_path) & a given folder contains the corresponding enhanced/output images (represented by output_folder_path)
# 2. calculate_psnr() calculate the psnr of each groundtruth-enhanced image pair, then append/accumulate/sum the psnr of all image pairs in psnr_values, then calculate the mean psnr using the appended psnr and total number of image pairs [(Mean psnr) = (Total psnr)/(Total number of image pairs)]

# 示例使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
psnr = PSNR(data_range=1.).to(device) # Performance metric: PSNR
ground_truth_folder_path = "D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/test_data/DICM"
output_folder_path = "D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/result/DICM"
average_psnr_value = calculate_psnr(ground_truth_folder_path, output_folder_path)
print(f'Average PSNR: {average_psnr_value}')

# Insights: 
# 1) When using the saved original and enhanced images to calculate the PSNR, the PSNR will be the same every time.
# 2) When using the original and enhanced images which are not saved yet to calculate the PSNR, the PSNR will fluctuate a little bit every time. Why?


