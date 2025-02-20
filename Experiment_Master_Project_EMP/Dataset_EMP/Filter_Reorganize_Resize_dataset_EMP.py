# This script aims to take an original SICE dataset Part 2 to filter, reorganize and followed by resizing it.

import glob
import shutil
import os
import time
import csv
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

def check_lists(list1, list2): # Used to check if 2 lists have the same elements
    list1_sorted = sorted(list1)
    list2_sorted = sorted(list2)

    if list1_sorted == list2_sorted:
        return True
    else:
        return False
    

def reorganize(): # Used to reorganize the original dataset

    print("-------------------------Part 1: Filtering & Reorganizing dataset operations begins-------------------------\n")
# ** Save all the files according to 2 subfolders only, "Input" OR "Label"
    # Take the target dataset from the source path, then save it in the destination path in the rearranged way
    folder_list = os.listdir(config.src_dir) # Return a list of folder names available in the root directory specified by src_dir.
    # print("Folder list:", folder_list)
    sorted_folder_list = sorted(folder_list, key=len) # sort the folder names in the list called folder_list based on the length of name
    # print("\nSorted folder list:", sorted_folder_list)
    # print("\n")
    
    # Create a csv file to record the metadata of the rearranged dataset
    current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
    csv_result_filename = 'Metadata-Reorganized-' + config.source_dataset_name + '-All' + '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
    csv_result_filepath = os.path.join(config.dst_dir,csv_result_filename) # Create the path to the csv that stores the metrics data
    with open(csv_result_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
        writer = csv.writer(csvfile, delimiter=',') # Create a csv writer. From the list of strings called title, the delimiter split the string after a "," into an element
        title = ['Source dataset name:', config.source_dataset_name, 'Partition name', 'All', 'Date and Time:', current_date_time_string]
        writer.writerow(title) # The writer writes the list of strings called "title"
        writer.writerow([]) # The writer writes a blank new line
        writer.writerow(['The sorted folder list:'])
        writer.writerow(sorted_folder_list) 
        writer.writerow([])
        writer = csv.DictWriter(csvfile, fieldnames=metadata.keys()) # Create a new csv writer. The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
        writer.writeheader() # The writer writes the header on the csv file

    for folder_name in tqdm(sorted_folder_list, desc="Filtering & Reorganizing original dataset"): # For each folder in the sorted folder list. tqdm is used to create the progress bar.
        if folder_name != "Label": 
        # if folder_name == "92": 
            file_list = glob.glob(config.src_dir+"/"+folder_name+"/*") # Returns a list of files' absolute path (of any extension) that are inside the specified path (src_dir+"/"+folder_name+"/*")
            metadata['folder_name'] = folder_name # Current folder name
            metadata['files_num_for_the_folder'] = len(file_list)
            # print("folder_name:", metadata['folder_name'])
            # print("file_num:", metadata['files_num_for_the_folder'])
            # print("\nFolder name:" + metadata['folder_name'] + "; Number of files in this folder:" + str(metadata['files_num_for_the_folder']) + "\n" ) # Show the current folder name and the number of files available in it
            
            # if metadata['files_num_for_the_folder'] == 7: # Retain the first 3 images and remove the remaining images if there are 7 images in the folder
            #     for index, file in enumerate(file_list):
            #         if index <= 2: # Only copy the 0th, 1st, and 2nd images  (total 3 images)
            # # for file in file_list: # For each image in the file list (available inside the current folder)
            #             metadata['image_counter'] += 1 # Increase the image counter by 1
                        
            #             # Part 1: Input image
            #             shutil.copy2(file, temp_destination_Input_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_Input folder inside the Input folder, in the destination root folder. # Argument1: Absolute path of the file; Argument2: Absolute path of the folder/destination that will save the file. References: https://expertbeacon.com/python-copy-file-copying-files-to-another-directory/
            #             Temp_Input_file, = glob.glob(temp_destination_Input_folder_path + "/*") # Match every file and folder from a given folder
            #             # print("File in Temp_Input_file: ",Temp_Input_file)
            #             Temp_Input_filename, Temp_Input_file_extension = os.path.splitext(Temp_Input_file) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext
            #             new_name_Input_file = str(metadata['image_counter']).zfill(4) + Temp_Input_file_extension # Rename the file with the current image counter and its original file extension. ".zfill(4)" will perform zero-padding for str having length (character numbers) less than 4, else the str will be returned as it is. Ref:https://note.nkmk.me/en/python-zero-padding/
            #             shutil.move(Temp_Input_file, destination_Input_folder_path + "/" + new_name_Input_file ) # Move the renamed file from Temp_Input folder out to the Input folder, in the destination root folder
                        
            #             # Part 2: Ground truth image
            #             GroundTruth_file, = glob.glob(GroundTruth_src_dir + "/" + folder_name + ".*") # Use sequence unpacking method to get the only element from the list. Ref: https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python
            #             # print(GroundTruth_file)
            #             shutil.copy2(GroundTruth_file, temp_destination_GroundTruth_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_GroundTruth folder inside the Ground Truth folder, in the destination root folder
            #             Temp_GroundTruth_file, = glob.glob(temp_destination_GroundTruth_folder_path + "/" + folder_name + ".*") # Get the absolute path of the file available in the Temp_GroundTruth folder, in the destination root folder
            #             # print(Temp_GroundTruth_file)
            #             Temp_GroundTruth_filename, Temp_GroundTruth_file_extension = os.path.splitext(Temp_GroundTruth_file) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext
            #             # print(Temp_GroundTruth_file_extension)
            #             new_name_GroundTruth_file = str(metadata['image_counter']).zfill(4) + Temp_GroundTruth_file_extension # Rename the file with the current image counter and its original file extension
            #             shutil.move(Temp_GroundTruth_file, destination_GroundTruth_folder_path + "/" + new_name_GroundTruth_file ) # Move the renamed file from Temp_GroundTruth folder out to the GroundTruth folder, in the destination root folder
                

            #             # Part 3: Update the metadata in the csv file
            #             with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
            #                 writer = csv.DictWriter(csvfile, fieldnames=metadata.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
            #                 writer.writerow(metadata) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file

            # if metadata['files_num_for_the_folder'] == 9: # Retain the first 3 images and remove the remaining images if there are 7 images in the folder
            #     for index, file in enumerate(file_list):
            #         if index <= 3: # Only copy the 0th, 1st, 2nd and 3rd images (total 4 images)
            # # for file in file_list: # For each image in the file list (available inside the current folder)
            #             metadata['image_counter'] += 1 # Increase the image counter by 1
                        
            #             # Part 1: Input image
            #             shutil.copy2(file, temp_destination_Input_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_Input folder inside the Input folder, in the destination root folder. # Argument1: Absolute path of the file; Argument2: Absolute path of the folder/destination that will save the file. References: https://expertbeacon.com/python-copy-file-copying-files-to-another-directory/
            #             Temp_Input_file, = glob.glob(temp_destination_Input_folder_path + "/*") # Match every file and folder from a given folder
            #             # print("File in Temp_Input_file: ",Temp_Input_file)
            #             Temp_Input_filename, Temp_Input_file_extension = os.path.splitext(Temp_Input_file) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext
            #             new_name_Input_file = str(metadata['image_counter']).zfill(4) + Temp_Input_file_extension # Rename the file with the current image counter and its original file extension. ".zfill(4)" will perform zero-padding for str having length (character numbers) less than 4, else the str will be returned as it is. Ref:https://note.nkmk.me/en/python-zero-padding/
            #             shutil.move(Temp_Input_file, destination_Input_folder_path + "/" + new_name_Input_file ) # Move the renamed file from Temp_Input folder out to the Input folder, in the destination root folder
                        
            #             # Part 2: Ground truth image
            #             GroundTruth_file, = glob.glob(GroundTruth_src_dir + "/" + folder_name + ".*") # Use sequence unpacking method to get the only element from the list. Ref: https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python
            #             # print(GroundTruth_file)
            #             shutil.copy2(GroundTruth_file, temp_destination_GroundTruth_folder_path) # Copy the ground truth image from source folder, then paste it to the Temp_GroundTruth folder inside the Ground Truth folder, in the destination root folder
            #             Temp_GroundTruth_file, = glob.glob(temp_destination_GroundTruth_folder_path + "/" + folder_name + ".*") # Get the absolute path of the file available in the Temp_GroundTruth folder, in the destination root folder
            #             # print(Temp_GroundTruth_file)
            #             Temp_GroundTruth_filename, Temp_GroundTruth_file_extension = os.path.splitext(Temp_GroundTruth_file) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext
            #             # print(Temp_GroundTruth_file_extension)
            #             new_name_GroundTruth_file = str(metadata['image_counter']).zfill(4) + Temp_GroundTruth_file_extension # Rename the file with the current image counter and its original file extension
            #             shutil.move(Temp_GroundTruth_file, destination_GroundTruth_folder_path + "/" + new_name_GroundTruth_file ) # Move the renamed file from Temp_GroundTruth folder out to the GroundTruth folder, in the destination root folder
                

            #             # Part 3: Update the metadata in the csv file
            #             with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
            #                 writer = csv.DictWriter(csvfile, fieldnames=metadata.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and IQA_metrics_data.keys() as the elements=keys of the header
            #                 writer.writerow(metadata) # The writer writes the data (value) of IQA_metrics_data dictionary in sequence as a row on the csv file

            if metadata['files_num_for_the_folder'] <= 6: # Retain the first 2 image and remove the remaining images if there are at most 6 images in the folder
                for index, file in enumerate(file_list):
                    if index <= 1: # Only copy the 0th and 1st image  (total 2 images)
            # for file in file_list: # For each image in the file list (available inside the current folder)
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

            elif metadata['files_num_for_the_folder'] <= 8: # Retain the first 3 images and remove the remaining images if there are 6<images<=8  in the folder
                for index, file in enumerate(file_list):
                    if index <= 2: # Only copy the 0th, 1st and 2nd images (total 3 images)
            # for file in file_list: # For each image in the file list (available inside the current folder)
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

            elif metadata['files_num_for_the_folder'] <= 9: # Retain the first 4 images and remove the remaining images if there are 8<images<=9 images in the folder
                for index, file in enumerate(file_list):
                    if index <= 3: # Only copy the 0th, 1st, 2nd and 3rd images (total 4 images)
            # for file in file_list: # For each image in the file list (available inside the current folder)
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

    print("\n-------------------------Reorganizing dataset operations status-------------------------")
    print("Operations completed, Summary:")
    print("Total number of images in Input folder [Filtered & Reorganized dataset]:", total_num_Input_image)
    print("Total number of images in Ground Truth folder [Filtered & Reorganized dataset]:", total_num_GroundTruth_image)    

    with open(csv_result_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of IQA_metrics_data dictionary to that csv file.
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([])
        writer.writerow(['-------------------------Reorganizing dataset operations status-------------------------'])
        writer.writerow(['Operations completed, Summary:'])
        summary = ['Total number of images in Input folder [Reorganized dataset]:', total_num_Input_image, 'Total number of images in Ground Truth folder [Reorganized dataset]:', total_num_GroundTruth_image]
        writer.writerow(summary) # The writer writes the list of strings called "title"
        file_Input_filename_list = []
        file_GroundTruth_filename_list = []
        for file_Input in os.listdir(destination_Input_folder_path): # Get a list of filename (file extension) of Input files
            file_Input_filename, file_Input_extension = os.path.splitext(file_Input) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext             
            file_Input_filename_list.append(file_Input_filename)

        for file_GroundTruth in os.listdir(destination_GroundTruth_folder_path): # Get a list of filename (file extension) of GroundTruth files
            file_GroundTruth_filename, file_GroundTruth_extension = os.path.splitext(file_GroundTruth) # Get an absolute path of a file and split it into path and file extension respectively. Ref:https://docs.python.org/3/library/os.path.html#os.path.splitext             
            file_GroundTruth_filename_list.append(file_GroundTruth_filename)

        # print("file_Input_filename_list:", file_Input_filename_list) # print all the filenames (excluding file extension) of Input files
        # print("file_GroundTruth_filename_list:", file_GroundTruth_filename_list) # print all the filenames (excluding file extension) of GroundTruth files

        if check_lists(file_Input_filename_list, file_GroundTruth_filename_list): # check if original dataset Input folder and GroundTruth folder have the same elements
            print("The original dataset Input folder and GroundTruth folder have the same elements")
            writer.writerow(['The original dataset Input folder and GroundTruth folder have the same elements'])
        else:
            print("The original dataset Input folder and GroundTruth folder do not have the same elements")
            writer.writerow(['The original dataset Input folder and GroundTruth folder do not have the same elements'])

        # if check_lists(os.listdir(destination_Input_folder_path), os.listdir(destination_GroundTruth_folder_path)): # check if original dataset Input folder and GroundTruth folder have the same elements
        #     print("The original dataset Input folder and GroundTruth folder have the same elements")
        #     writer.writerow(['The original dataset Input folder and GroundTruth folder have the same elements'])
        # else:
        #     print("The original dataset Input folder and GroundTruth folder do not have the same elements")
        #     writer.writerow(['The original dataset Input folder and GroundTruth folder do not have the same elements'])
        

def resize():
    
    print("-------------------------Part 2:Resizing dataset operations begins-------------------------\n")

    folderpaths_to_resize = [destination_Input_folder_path, destination_GroundTruth_folder_path]
    foldernames_list_forRecordPurposeOnly = ['Input_Folder', 'GroundTruth_Folder']
    foldername_index_forRecordPurposeOnly = 0

    for folder in folderpaths_to_resize:
        for file in tqdm(glob.iglob(folder + "/*"), desc= f"Resizing images in {foldernames_list_forRecordPurposeOnly[foldername_index_forRecordPurposeOnly]}"): # For each image in the file list (available inside the current folder)
            
            ori_image = Image.open(file) # Open the image located at the given image's absolute path, using PIL. That image has the shape of [height,width,channels].
            if not ori_image.mode == 'RGB': # if the image is not a RGB image (using RGB color representation, RGB mode). Because .jpeg and .png are the most widely found graphic file formats on website. .jpeg only supports RGB mode, but .png can support multiple color modes including RGB and RGBA. While the image data tensors in this script are required to be in RGB mode (must have 3 color channels, as R,G,B respectively) [because the model's first convolutional layer is defined to take each sample/image of 3 channels(RGB image), for a batch of input samples/images], so we need to convert any color modes of image data into RGB mode. Color representation = color mode.
                ori_image = ori_image.convert('RGB') # convert the image data color representation into RGB (Red, Green, Blue) mode, More info: https://dnmtechs.com/converting-rgba-png-to-rgb-using-pil-in-python-3/
            resized_image = ori_image.resize((config.image_size_width, config.image_size_height), Image.LANCZOS) # resize the input image

            # resized_image.save(os.path.join(big_img_folder_path, os.path.basename(file))) # save resized images to a new folder, while remain its filename unchanged
            resized_image.save(os.path.join(folder, os.path.basename(file))) # save resized images to the existing folder, while remain its filename unchanged
        
        foldername_index_forRecordPurposeOnly += 1

    print("\n-------------------------Resizing dataset operations status-------------------------")
    print("Operations completed, Summary:")
    print(f'Each resized image has dimensions of (height x width, pixel dimensions): {config.image_size_height} x {config.image_size_width}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
    
    parser.add_argument('--src_dir', type=str, default= "C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Unprocessed/SICE_Dataset_Part2_EMP/Dataset_Part2") # The absolute path of the source dataset
    parser.add_argument('--dst_dir', type=str, default="C:/Master_XMUM_usages/AI_Master_New/Experiment_Master_Project_EMP/Dataset_EMP/Preprocessed_SICE_Dataset_Part2_EMP") # The absolute path of the destination root folder that saves the rearranged dataset
    parser.add_argument('--source_dataset_name', type=str, default="SICE_Dataset_Part2") # The name of the source dataset
    parser.add_argument('--Input_folder_basename', type=str, default= "/Input") # Base name for the Input folder for the train and validation datasets
    parser.add_argument('--GroundTruth_folder_basename', type=str, default= "/GroundTruth") # Base name for the GroundTruth folder for the train and validation datasets
    parser.add_argument('--image_size_height', type=int, default=900) # The height size of the input validation images to be resized (in pixel dimension)
    parser.add_argument('--image_size_width', type=int, default=1200) # The width size of the input validation images to be resized (in pixel dimension)
    parser.add_argument('--random_seed', type=int, default=42) # The seed value used to initialize the numpy random number generator for reproducible random number generations
    
    config = parser.parse_args() 

    # Create the dictionary to record the metadata of the reorganized dataset
    metadata = {'folder_name': 0, 'files_num_for_the_folder': 0, 'image_counter': -1}

    # Create the required directories for reorganize()
    os.makedirs(config.src_dir, exist_ok=True)
    destination_Input_folder_path = config.dst_dir+"/Input" # Create the absolute path of the folder/destination that will save the input files/images, in the destination root folder
    os.makedirs(destination_Input_folder_path, exist_ok=True) # Create the folder/destination that will save the input files/images, in the destination root folder
    destination_GroundTruth_folder_path = config.dst_dir+"/GroundTruth" # Create the absolute path of the folder/destination that will save the ground truth files/images, in the destination root folder
    os.makedirs(destination_GroundTruth_folder_path, exist_ok=True) # Create the folder/destination that will save the ground truth files/images, in the destination root folder
    GroundTruth_src_dir = config.src_dir+"/Label" # The absolute path of the folder that contains the ground truth files/images, in the source folder
    temp_destination_Input_folder_path = destination_Input_folder_path+"/Temp_Input" # The Temp_Input folder, in the destination root folder
    os.makedirs(temp_destination_Input_folder_path, exist_ok=True)
    temp_destination_GroundTruth_folder_path = destination_GroundTruth_folder_path+"/Temp_GroundTruth" # The Temp_GroundTruth folder, in the destination root folder
    os.makedirs(temp_destination_GroundTruth_folder_path, exist_ok=True)
    
    # Reorganize the original dataset
    # So rule of thumb of using self-defined functions, must have the return operator, but the function argument is optional.
    
    print("***********************************************************************************************")
    reorganize() # when using self-defined functions, must have return operator at the end to get the results obtained from the operations within the function. You only can get the results by using the return operator. But any changes made to a dictionary within the function will also be reflected outside the function, even without using return operator. 
    print("***********************************************************************************************\n")

    # print("***********************************************************************************************")
    # # Partition the original dataset into train dataset and validation dataset, with the predefined ratios
    # partition()
    # print("***********************************************************************************************\n")

    print("***********************************************************************************************")
    resize()
    print("***********************************************************************************************\n")
    
    print("-------------------------All operations completed-------------------------")

# Instruction: Use this script to reorganize, partition, and resize the original image dataset.



