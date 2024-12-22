# Obtain the script from: https://github.com/realnghon/Zero-DCE_Enhanced/blob/master/Zero-DCE_code/PSNR.py
# Each ground truth image and its corresponding enhanced/output image must have the same filename. But they can be located at different folders (different root directory).
# Peak Signal-to-Noise Ratio (PSNR) is defined as the ratio of the maximum possible signal value to the value of the noise that affects the fidelity of its representation.
# PSNR is a general metric for mentioning the quality of an image or any video stream. It aids in determining the degree of information loss or degradation brought on by compression. The apparent quality of the compressed image or video increases as PSNR increases because distortion becomes less noticeable.

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread
import numpy as np
import os

def calculate_psnr(img1, img2):
    img1 = imread(img1, as_gray=True) # load the ground truth image whose path=os.path.join(dir1, file), then save as variable img1
    img2 = imread(img2, as_gray=True) # load the enhanced/output image whose path=os.path.join(dir2, file), then save as variable img2
    
    # Resize images to the smallest one's shape
    img1 = np.resize(img1, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
    img2 = np.resize(img2, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))

    value = psnr(img1, img2) # Calculate the PSNR (a function of scikit-image) on this groundtruth-enhanced images pair. skimage.metrics.peak_signal_noise_ratio only calculates the PSNR of an image pair, in contrast to torchmetrics.PeakSignalNoiseRatio.
    
    return value

# Concept/Operating principle of calculating psnr in this script [performed by calculate_psnr()]:
# 1. For a given folder contains ground truth images (represented by ground_truth_folder_path) & a given folder contains the corresponding enhanced/output images (represented by output_folder_path)
# 2. calculate_psnr() calculate the psnr of each groundtruth-enhanced image pair, then append/accumulate/sum the psnr of all image pairs in psnr_values, then calculate the mean psnr using the appended psnr and total number of image pairs [(Mean psnr) = (Total psnr)/(Total number of image pairs)]

# 示例使用
ground_truth_folder_path = "D:/AI_Master_New/try_imagepairs/00756_normal_light.png"
# output_folder_path = "D:/AI_Master_New/try_imagepairs/00756_enhanced_ZeroDCE.png"
output_folder_path = "D:/AI_Master_New/try_imagepairs/00756_low_light_enhanced.jpg"
average_psnr_value = calculate_psnr(ground_truth_folder_path, output_folder_path)
print(f'Average PSNR: {average_psnr_value}')


