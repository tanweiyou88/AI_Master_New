# Obtain the script from: https://github.com/realnghon/Zero-DCE_Enhanced/blob/master/Zero-DCE_code/MAE.py
# Each ground truth image and its corresponding enhanced/output image must have the same filename. But they can be located at different folders (different root directory).
# MAE calculates the mean absolute error between the images (pixel-by-pixel).

from skimage.io import imread
import numpy as np
import os

def calculate_mae(dir1, dir2):
    # List comprehension part: For every f=filename in os.listdir(dir1)=(a list of filenames available in dir1), if os.path.isfile(os.path.join(dir1, f)=True=(if a filepath represented as os.path.join(dir1, f) exists), then add f=filename to a new list. After the list comprehension is implemented, only sort the elements/entries in that new list and then add them to the variable called files1.
    files1 = sorted([f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))])
    # List comprehension part: For every f=filename in os.listdir(dir2)=(a list of filenames available in dir2), if os.path.isfile(os.path.join(dir2, f)=True=(if a filepath represented as os.path.join(dir2, f) exists), then add f=filename to a new list. After the list comprehension is implemented, only sort the elements/entries in that new list and then add them to the variable called files2.
    files2 = sorted([f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))])
    mae_values = [] # create an empty list called mae_values

    for file in files1: # For each file=filename=(ground truth image) in files1=(the created new list of sorted filenames available in folder dir1). Each iteration will get a groundtruth-enhanced images pair.
        if file in files2: # Check if file=filename is present in files2=(the created new list of sorted filenames available in folder dir2). A True will be returned if it exists.
            # Get a groundtruth-enhanced images pair
            img1 = imread(os.path.join(dir1, file)) # load the ground truth image whose path=os.path.join(dir1, file), then save as variable img1
            img2 = imread(os.path.join(dir2, file)) # load the enhanced/output image whose path=os.path.join(dir2, file), then save as variable img2

            # 确保两个图片尺寸一致
            # If want to resize an image, it is easier to downsize an image than it is to enlarge an image. This is because when an image is enlarged, the photo editor must create and add new pixel information -- based on its best guesses (interpolation, where the software estimates the color of the new pixels based on the color of existing pixels. These algorithms, while sophisticated, can't invent details that aren't present in the original image) -- to achieve a larger size which typically results in either a very pixelated or very soft and blurry looking image. However, if an image is downsized, some pixels are required to be removed only, affecting the image quality lesser. However, both upsampling and downsampling image will lead to quality loss.  
            # That's why to ensure both the ground truth image and its corresponding enhanced/output image have the same dimensions/sizes, we use downsampling. So, if the images of the groundtruth-enhanced images pair have different dimensions, we resize both images respectively using the smallest height dimension and smallest width dimension.
            img1 = np.resize(img1, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            img2 = np.resize(img2, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))

            value = np.mean(np.abs(img1 - img2)) # Calculate the MAE=mean (a function of numpy) on this groundtruth-enhanced images pair
            mae_values.append(value) # Add this calculated MAE to the mae_values list, as a new element/entry

    # Usage of Ternary operator: If mae_values is NOT an empty list (at least 1 element/entry is available in the list), then execute np.mean(mae_values)=([Sum all elements/entries in the list called mae_values]/[Total number of groundtruth-enhanced images pair, represented as the number of elements/entries in in the list called mae_values]), then store the result in the variable average_mae. Else (if pnsr_values is an empty list [no element/entry is available in the list]), store the value 0 in the average_mae.
    average_mae = np.mean(mae_values) if mae_values else 0
    return average_mae

# 示例使用
ground_truth_folder_path = "D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/test_data/DICM"
output_folder_path = "D:/AI_Master_New/Low-Light_Image_Enhancement/Zero-DCE_code_LiChongYi_2021/data/result/DICM"
average_mae_value = calculate_mae(ground_truth_folder_path, output_folder_path)
print(f'Average MAE: {average_mae_value}')