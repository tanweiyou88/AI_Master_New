# This script aims to take a source dataset to reorganize it before saving it at a new destination folder

import glob
import shutil
import os
import time
import csv
import argparse
from tqdm import tqdm

def check_lists(list1, list2): # Used to check if 2 lists have the same elements
    list1_sorted = sorted(list1)
    list2_sorted = sorted(list2)

    if list1_sorted == list2_sorted:
        return True
    else:
        return False
    

def reorganize(config, metadata): # Used to reorganize the original dataset
# # ** Save all the files into their subfolders respectively **: Version 1
# for folder_name in sorted_folder_list: # For each folder in the sorted folder list
#     if folder_name != "Label": 
#         file_list = glob.glob(src_dir+"/"+folder_name+"/*") # Returns a list of files' absolute path (of any extension) that are inside the specified path (src_dir+"/"+folder_name+"/*")
#         print("Folder name:" + folder_name + "; Number of files in this folder:" + str(len(file_list)) ) # Show the current folder name and the number of files available in it
#         for file in file_list: # For each image in the file list (available inside the current folder)
#             destination_folder_path = dst_dir+"/"+folder_name # Create the absolute path of the folder/destination that will save the files, in the destination root folder
#             os.makedirs(destination_folder_path, exist_ok=True) # Create the folder/destination that will save the files, in the destination root folder
#             shutil.copy2(file, destination_folder_path) # Argument1: Absolute path of the file; Argument2: Absolute path of the folder/destination that will save the file. References: https://expertbeacon.com/python-copy-file-copying-files-to-another-directory/


# # ** Save all the files according to 2 subfolders only, "Input" OR "Label" **: Version 2
# destination_Input_folder_path = dst_dir+"/Input" # Create the absolute path of the folder/destination that will save the input files/images, in the destination root folder
# os.makedirs(destination_Input_folder_path, exist_ok=True) # Create the folder/destination that will save the input files/images, in the destination root folder
# destination_GroundTruth_folder_path = dst_dir+"/GroundTruth" # Create the absolute path of the folder/destination that will save the ground truth files/images, in the destination root folder
# os.makedirs(destination_GroundTruth_folder_path, exist_ok=True) # Create the folder/destination that will save the ground truth files/images, in the destination root folder
# GroundTruth_src_dir = src_dir+"/Label" # The absolute path of the folder that contains the ground truth files/images, in the source folder
# temp_destination_GroundTruth_folder_path = destination_GroundTruth_folder_path+"/Temp_GroundTruth"
# os.makedirs(temp_destination_GroundTruth_folder_path, exist_ok=True)

# for folder_name in sorted_folder_list: # For each folder in the sorted folder list
#     # if folder_name != "Label": 
#     if folder_name == "92": 
#         file_list = glob.glob(src_dir+"/"+folder_name+"/*") # Returns a list of files' absolute path (of any extension) that are inside the specified path (src_dir+"/"+folder_name+"/*")
#         file_num = len(file_list)
#         print("Folder name:" + folder_name + "; Number of files in this folder:" + str(file_num) ) # Show the current folder name and the number of files available in it
#         for file in file_list: # For each image in the file list (available inside the current folder)
#             # destination_input_folder_path = dst_dir+"/Input" # Create the absolute path of the folder/destination that will save the files, in the destination root folder
#             # os.makedirs(destination_input_folder_path, exist_ok=True) # Create the folder/destination that will save the files, in the destination root folder
#             shutil.copy2(file, destination_Input_folder_path) # Argument1: Absolute path of the file; Argument2: Absolute path of the folder/destination that will save the file. References: https://expertbeacon.com/python-copy-file-copying-files-to-another-directory/
#         for i in range(file_num):
#             # GroundTruth_file = GroundTruth_src_dir  + "/" + folder_name
#             GroundTruth_file, = glob.glob(GroundTruth_src_dir + "/" + folder_name + ".*") # Use sequence unpacking method to get the only element from the list. Ref: https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python
#             print(GroundTruth_file)
#             # GroundTruth_file = GroundTruth_file.replace(folder_name,str(i))
#             shutil.copy2(GroundTruth_file, temp_destination_GroundTruth_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_GroundTruth folder inside the Ground Truth folder, in the destination root folder
#             Temp_GroundTruth_file, = glob.glob(temp_destination_GroundTruth_folder_path + "/" + folder_name + ".*") # Get the absolute path of the file available in the Temp_GroundTruth folder, in the destination root folder
#             print(Temp_GroundTruth_file)
#             renamed_GroundTruth_file = str(i) + ".jpg"
#             shutil.move(Temp_GroundTruth_file, destination_GroundTruth_folder_path + "/" + renamed_GroundTruth_file )
#             # os.rename(destination_GroundTruth_folder_path + "/" , dest)
#             # Temp_GroundTruth_file = Temp_GroundTruth_file.replace(folder_name,str(i)) # In the Temp folder, replace the part of the absolute path of the file that having "folder_name" with "str(i)" 
#             # shutil.copy2(Temp_GroundTruth_file, destination_GroundTruth_folder_path) # Copy the renamed ground truth image from the Temp_GroundTruth folder to the Ground Truth folder, in the destination root folder
#         shutil.rmtree(temp_destination_GroundTruth_folder_path) # Remove the Temp_GroundTruth folder, in the destination root folder


# ** Save all the files according to 2 subfolders only, "Input" OR "Label" **: Version 3, Use This
    # Create the required directories
    print("-------------------------Reorganizing dataset operations begins-------------------------")
    os.makedirs(config.src_dir, exist_ok=True)
    destination_Input_folder_path = config.dst_dir+"/Input_All" # Create the absolute path of the folder/destination that will save the input files/images, in the destination root folder
    os.makedirs(destination_Input_folder_path, exist_ok=True) # Create the folder/destination that will save the input files/images, in the destination root folder
    destination_GroundTruth_folder_path = config.dst_dir+"/GroundTruth_All" # Create the absolute path of the folder/destination that will save the ground truth files/images, in the destination root folder
    os.makedirs(destination_GroundTruth_folder_path, exist_ok=True) # Create the folder/destination that will save the ground truth files/images, in the destination root folder
    GroundTruth_src_dir = config.src_dir+"/Label" # The absolute path of the folder that contains the ground truth files/images, in the source folder
    temp_destination_Input_folder_path = destination_Input_folder_path+"/Temp_Input" # The Temp_Input folder, in the destination root folder
    os.makedirs(temp_destination_Input_folder_path, exist_ok=True)
    temp_destination_GroundTruth_folder_path = destination_GroundTruth_folder_path+"/Temp_GroundTruth" # The Temp_GroundTruth folder, in the destination root folder
    os.makedirs(temp_destination_GroundTruth_folder_path, exist_ok=True)

    # Take the target dataset from the source path, then save it in the destination path in the rearranged way
    folder_list = os.listdir(config.src_dir) # Return a list of folder names available in the root directory specified by src_dir.
    print("Folder list:", folder_list)
    sorted_folder_list = sorted(folder_list, key=len) # sort the folder names in the list called folder_list based on the length of name
    print("Sorted folder list:", sorted_folder_list)
    
    # Create a csv file to record the metadata of the rearranged dataset
    current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
    csv_result_filename = 'Metadata-Reorganized-' + config.source_dataset_name + '-' + config.partition_name + '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
    csv_result_filepath = os.path.join(config.dst_dir,csv_result_filename) # Create the path to the csv that stores the metrics data
    with open(csv_result_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
        writer = csv.writer(csvfile, delimiter=',') # Create a csv writer. From the list of strings called title, the delimiter split the string after a "," into an element
        title = ['Source dataset name:', config.source_dataset_name, 'Partition name', config.partition_name, 'Date and Time:', current_date_time_string]
        writer.writerow(title) # The writer writes the list of strings called "title"
        writer.writerow([]) # The writer writes a blank new line
        writer.writerow(['The sorted folder list:'])
        writer.writerow(sorted_folder_list) 
        writer.writerow([])
        writer = csv.DictWriter(csvfile, fieldnames=metadata.keys()) # Create a new csv writer. The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
        writer.writeheader() # The writer writes the header on the csv file

    

    for folder_name in tqdm(sorted_folder_list, desc="Reorganizing original dataset"): # For each folder in the sorted folder list. tqdm is used to create the progress bar.
        if folder_name != "Label": 
        # if folder_name == "92": 
            file_list = glob.glob(config.src_dir+"/"+folder_name+"/*") # Returns a list of files' absolute path (of any extension) that are inside the specified path (src_dir+"/"+folder_name+"/*")
            metadata['folder_name'] = folder_name # Current folder name
            metadata['files_num_for_the_folder'] = len(file_list)
            print("\n\nFolder name:" + metadata['folder_name'] + "; Number of files in this folder:" + str(metadata['files_num_for_the_folder']) ) # Show the current folder name and the number of files available in it
            for file in file_list: # For each image in the file list (available inside the current folder)
                metadata['image_counter'] += 1 # Increase the image counter by 1
                
                # Part 1: Input image
                shutil.copy2(file, temp_destination_Input_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_Input folder inside the Input folder, in the destination root folder. # Argument1: Absolute path of the file; Argument2: Absolute path of the folder/destination that will save the file. References: https://expertbeacon.com/python-copy-file-copying-files-to-another-directory/
                Temp_Input_file, = glob.glob(temp_destination_Input_folder_path + "/*") # Match every file and folder from a given folder
                # print("File in Temp_Input_file: ",Temp_Input_file)
                Temp_Input_filename, Temp_Input_file_extension = os.path.splitext(Temp_Input_file) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext
                new_name_Input_file = str(metadata['image_counter']).zfill(4) + Temp_Input_file_extension # Rename the file with the current image counter and its original file extension. ".zfill(4)" will perform zero-padding for str having length (character numbers) less than 4, else the str will be returned as it is. Ref:https://note.nkmk.me/en/python-zero-padding/
                shutil.move(Temp_Input_file, destination_Input_folder_path + "/" + new_name_Input_file ) # Move the renamed file from Temp_Input folder out to the Input folder, in the destination root folder
                
                # Part 2: Ground truth image
                GroundTruth_file, = glob.glob(GroundTruth_src_dir + "/" + folder_name + ".*") # Use sequence unpacking method to get the only element from the list. Ref: https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python
                # print(GroundTruth_file)
                shutil.copy2(GroundTruth_file, temp_destination_GroundTruth_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_GroundTruth folder inside the Ground Truth folder, in the destination root folder
                Temp_GroundTruth_file, = glob.glob(temp_destination_GroundTruth_folder_path + "/" + folder_name + ".*") # Get the absolute path of the file available in the Temp_GroundTruth folder, in the destination root folder
                # print(Temp_GroundTruth_file)
                Temp_GroundTruth_filename, Temp_GroundTruth_file_extension = os.path.splitext(Temp_GroundTruth_file) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext
                # print(Temp_GroundTruth_file_extension)
                new_name_GroundTruth_file = str(metadata['image_counter']).zfill(4) + Temp_GroundTruth_file_extension # Rename the file with the current image counter and its original file extension
                shutil.move(Temp_GroundTruth_file, destination_GroundTruth_folder_path + "/" + new_name_GroundTruth_file ) # Move the renamed file from Temp_GroundTruth folder out to the GroundTruth folder, in the destination root folder
                

                # Part 3: Update the metadata in the csv file
                with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
                    writer = csv.DictWriter(csvfile, fieldnames=metadata.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
                    writer.writerow(metadata) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file
                
    shutil.rmtree(temp_destination_Input_folder_path) # Remove the Temp_Input folder, in the destination root folder
    shutil.rmtree(temp_destination_GroundTruth_folder_path) # Remove the Temp_GroundTruth folder, in the destination root folder  

    # Generate summary of the reorganized dataset
    total_num_Input_image = len(os.listdir(destination_Input_folder_path) )
    total_num_GroundTruth_image = len(os.listdir(destination_GroundTruth_folder_path) )

    print("-------------------------Reorganizing dataset operations status-------------------------")
    print("Operations completed, Summary:")
    print("Total number of images in Input folder [Reorganized dataset]:", total_num_Input_image)
    print("Total number of images in Ground Truth folder [Reorganized dataset]:", total_num_GroundTruth_image)

    with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([])
        writer.writerow(['-------------------------Reorganizing dataset operations status-------------------------'])
        writer.writerow(['Operations completed, Summary:'])
        summary = ['Total number of images in Input folder [Reorganized dataset]:', total_num_Input_image, 'Total number of images in Ground Truth folder [Reorganized dataset]:', total_num_GroundTruth_image]
        writer.writerow(summary) # The writer writes the list of strings called "title"
        if check_lists(os.listdir(destination_Input_folder_path), os.listdir(destination_GroundTruth_folder_path)): # check if original dataset Input folder and GroundTruth folder have the same elements
            print("The original dataset Input folder and GroundTruth folder have the same elements\n")
            writer.writerow(['The original dataset Input folder and GroundTruth folder have the same elements'])
        else:
            print("The original dataset Input folder and GroundTruth folder do not have the same elements\n")
            writer.writerow(['The original dataset Input folder and GroundTruth folder do not have the same elements'])
        
    

def main(config):
        
        # Create the dictionary to record the metadata of the reorganized dataset
        metadata = {'folder_name': 0, 'files_num_for_the_folder': 0, 'image_counter': -1}
        
        # So rule of thumb of using self-defined functions, must have the return operator, but the function argument is optional.
        reorganize(config, metadata) # when using self-defined functions, must have return operator at the end to get the results obtained from the operations within the function. You only can get the results by using the return operator. But any changes made to a dictionary within the function will also be reflected outside the function, even without using return operator. 



if __name__ == '__main__':

    parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
    # Input Parameters
    parser.add_argument('--src_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/sampling_SICE_Part1_TrainingSet") # The absolute path of the source dataset
    parser.add_argument('--dst_dir', type=str, default="D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/Reorganized_sampling_SICE_Part1_TrainingSet") # The absolute path of the destination root folder that saves the rearranged dataset
    parser.add_argument('--source_dataset_name', type=str, default="SICE_Dataset_Part1") # The name of the source dataset
    parser.add_argument('--partition_name', type=str, default="All") # The name of the partition (All/Train/Validation)
    # parser.add_argument('--lr', type=float, default=0.0001) # Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
    # parser.add_argument('--weight_decay', type=float, default=0.0001)
    config = parser.parse_args() 

    main(config)
    
    
    