# Obtain the script from: https://github.com/realnghon/Zero-DCE_Enhanced/blob/master/Zero-DCE_code/SSIM.py
# Each ground truth image and its corresponding enhanced/output image must have the same filename. But they can be located at different folders (different root directory).
# Structural Similarity Index Measure (SSIM) quantifies image similarity by combining these three factors — luminance, contrast, and structure — into a single metric.
# Unlike PSNR, which focuses solely on pixel-by-pixel differences, SSIM considers/measures the structural information in an image, making it a more sophisticated metric that aligns better with human visual perception.

from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

def calculate_ssim(img1, img2):
    img1 = imread(img1, as_gray=True) # load the ground truth image whose path=os.path.join(dir1, file), then save as variable img1
    img2 = imread(img2, as_gray=True) # load the enhanced/output image whose path=os.path.join(dir2, file), then save as variable img2
    
    # Resize images to the smallest one's shape
    # If want to resize an image, it is easier to downsize an image than it is to enlarge an image. This is because when an image is enlarged, the photo editor must create and add new pixel information -- based on its best guesses (interpolation, where the software estimates the color of the new pixels based on the color of existing pixels. These algorithms, while sophisticated, can't invent details that aren't present in the original image) -- to achieve a larger size which typically results in either a very pixelated or very soft and blurry looking image. However, if an image is downsized, some pixels are required to be removed only, affecting the image quality lesser. However, both upsampling and downsampling image will lead to quality loss.  
    # That's why to ensure both the ground truth image and its corresponding enhanced/output image have the same dimensions/sizes, we use downsampling. So, if the images of the groundtruth-enhanced images pair have different dimensions, we resize both images respectively using the smallest height dimension and smallest width dimension.
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])

    img1_resized = resize(img1, (height, width), anti_aliasing=True)
    img2_resized = resize(img2, (height, width), anti_aliasing=True)

    # Calculate the ssim (a function of scikit-image) on this groundtruth-enhanced images pair
    value = ssim(img1_resized, img2_resized, data_range=img2_resized.max() - img2_resized.min())
    

    return value

# Example usage
ground_truth_folder_path = "D:/AI_Master_New/try_imagepairs/00756_normal_light.png"
# output_folder_path = "D:/AI_Master_New/try_imagepairs/00756_enhanced_ZeroDCE.png"
output_folder_path = "D:/AI_Master_New/try_imagepairs/00756_low_light_enhanced.jpg"
average_ssim = calculate_ssim(ground_truth_folder_path, output_folder_path)
print(f'Average SSIM: {average_ssim}')

