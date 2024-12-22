# This script aims to partition the files in a single folder into 2 folders, with the predefined ratio. Ref: https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified

import os
import numpy as np
import shutil
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
 

def partitioning(config): # Used to partition the original dataset into train dataset and validation dataset, with the predefined ratios
    print("-------------------------Partitioning dataset operations begins-------------------------\n")
    # random_seed = 42 # The seed value used to initialize the numpy random number generator for reproducible random number generations
    # val_ratio = 0.1 # The ratio of validation data from the original dataset
    val_ratio = config.val_ratio # The ratio of validation data from the original dataset
    train_ratio = 1-val_ratio # The ratio of train data from the original dataset

    # Part1: Create the required directories
    # Create the Input and GroundTruth folders for the train and validataion datasets respectively
    # root_dir = 'D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/Reorganized_sampling_SICE_Part1_TrainingSet' # The absolute path of the root folder
    # input = '/Input' # Base name for the Input folder
    # groundtruth = '/GroundTruth' # Base name for the GroundTruth folder

    train_Input_folder_path = config.root_dir +'/train' + config.Input_folder_basename
    os.makedirs(train_Input_folder_path, exist_ok=True) # create the Input folder for train dataset
    train_GroundTruth_folder_path = config.root_dir +'/train' + config.GroundTruth_folder_basename
    os.makedirs(train_GroundTruth_folder_path, exist_ok=True) # create the GroundTruth folder for train dataset
    val_Input_folder_path = config.root_dir +'/val' + config.Input_folder_basename
    os.makedirs(val_Input_folder_path, exist_ok=True) # create the Input folder for validation dataset
    val_GroundTruth_folder_path = config.root_dir +'/val' + config.GroundTruth_folder_basename
    os.makedirs(val_GroundTruth_folder_path, exist_ok=True) # create the GroundTruth folder for validation dataset

    # Create the absolute paths that reach to the Input and GroundTruth folders of the original dataset respectively 
    src_Input_folder_path = config.root_dir + '/Input_All' # The folder that stores the original dataset Input images
    src_GroundTruth_folder_path = config.root_dir + '/GroundTruth_All' # The folder that stores the original dataset GroundTruth images 

    # Part2: Split the files in the Input and GroundTruth folders of the original dataset into 2 partitions (train dataset & validation dataset) respectively, after shuffling
    # For Input images part:
    src_Input_FileNames = os.listdir(src_Input_folder_path) # Get a list of the original dataset Input image filenames
    np.random.seed(config.random_seed) # Set the random seed for the random number generator (RNG) in np.random, so that np.random will generate the same numbers for reproducible results
    np.random.shuffle(src_Input_FileNames) # Shuffle the original dataset Input image filenames in the list
    train_Input_FileNames, val_Input_FileNames,_ = np.split(np.array(src_Input_FileNames),
                                                            [int(len(src_Input_FileNames)*train_ratio), int(len(src_Input_FileNames)*1.0)]) # Split the first train_ratio filenames in the original dataset Input image filename list into train_FileNames as the list of train dataset Input image filenames, split the remaining original dataset Input image filenames in the list into val_FileNames as the list of validation dataset Input image filenames

    train_Input_FilePaths = [src_Input_folder_path+'/'+ name for name in train_Input_FileNames.tolist()] # For each train dataset Input image filename in the list, update it with its absolute path in the original dataset
    val_Input_FilePaths = [src_Input_folder_path+'/' + name for name in val_Input_FileNames.tolist()] # For each validation dataset Input image filename in the list, update it with its absolute path in the original dataset

    # Copy-pasting images
    for train_Input_FilePath in tqdm(train_Input_FilePaths, desc="Copying and pasting train dataset Input images"): # tqdm is used to create the progress bar.
        # print("Filenames of Input images of train dataset:",train_Input_FilePath)
        shutil.copy2(train_Input_FilePath, train_Input_folder_path) # Copy each train dataset Input image from the original dataset Input folder to the train dataset Input folder

    for val_Input_FilePath in tqdm(val_Input_FilePaths, desc="Copying and pasting validation dataset Input images"):
        shutil.copy2(val_Input_FilePath, val_Input_folder_path) # Copy each validation dataset Input image from the original dataset Input folder to the validation dataset Input folder


    # For GroundTruth images part:
    src_GroundTruth_FileNames = os.listdir(src_GroundTruth_folder_path) # Get a list of the original dataset Input image filenames
    np.random.seed(config.random_seed) # Set the random seed for the random number generator (RNG) in np.random, so that np.random will generate the same numbers for reproducible results
    np.random.shuffle(src_GroundTruth_FileNames) # Shuffle the original dataset Input image filenames in the list
    train_GroundTruth_FileNames, val_GroundTruth_FileNames,_ = np.split(np.array(src_GroundTruth_FileNames),
                                                            [int(len(src_GroundTruth_FileNames)*train_ratio), int(len(src_GroundTruth_FileNames)*1.0)]) # Split the first train_ratio filenames in the original dataset Input image filename list into train_FileNames as the list of train dataset Input image filenames, split the remaining original dataset Input image filenames in the list into val_FileNames as the list of validation dataset Input image filenames

    train_GroundTruth_FilePaths = [src_GroundTruth_folder_path+'/'+ name for name in train_GroundTruth_FileNames.tolist()] # For each train dataset Input image filename in the list, update it with its absolute path in the original dataset
    val_GroundTruth_FilePaths = [src_GroundTruth_folder_path+'/' + name for name in val_GroundTruth_FileNames.tolist()] # For each validation dataset Input image filename in the list, update it with its absolute path in the original dataset

    # Copy-pasting images
    for train_GroundTruth_FilePath in tqdm(train_GroundTruth_FilePaths, desc="Copying and pasting train dataset GroundTruth images"):
        # print("Filenames of Input images of train dataset:",train_Input_FilePath)
        shutil.copy2(train_GroundTruth_FilePath, train_GroundTruth_folder_path) # Copy each train dataset Input image from the original dataset Input folder to the train dataset Input folder

    for val_GroundTruth_FilePath in tqdm(val_GroundTruth_FilePaths, desc="Copying and pasting validation dataset GroundTruth images"):
        shutil.copy2(val_GroundTruth_FilePath, val_GroundTruth_folder_path) # Copy each validation dataset Input image from the original dataset Input folder to the validation dataset Input folder

    #Part3: Generate summary
    print("\n-------------------------Partitioning dataset operations status-------------------------")
    print("Operations completed, Summary:")
    print(f'Ratio:\n(Train dataset:Validation dataset) = ({train_ratio}:{val_ratio})\n\nTotal number of:')
    print('Original dataset Input images:', len(src_Input_FileNames)) # Get the total number of the original dataset Input images
    print('Original dataset GroundTruth images:', len(src_GroundTruth_FileNames)) # Get the total number of the original dataset GroundTruth images
    print('Train dataset Input images:', len(train_Input_FilePaths)) # Get the total number of the train dataset Input images 
    print('Train dataset GroundTruth images:', len(train_GroundTruth_FilePaths)) # Get the total number of the train dataset GroundTruth images 
    print('Validation dataset Input images:', len(val_Input_FilePaths)) # Get the total number of the validation dataset Input images
    print('Validation dataset GroundTruth images:', len(val_GroundTruth_FilePaths)) # Get the total number of the validation dataset GroundTruth images


    # Initialize CSV files
    # CSV file to record metadata of train dataset
    current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
    csv_result_train_filename = 'Metadata-Reorganized-' + config.source_dataset_name + '-' + 'Train' + '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
    csv_result_train_filepath = os.path.join(config.root_dir,csv_result_train_filename) # Create the path to the csv that stores the metrics data
    with open(csv_result_train_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
        writer = csv.writer(csvfile, delimiter=',') # Create a csv writer. From the list of strings called title, the delimiter split the string after a "," into an element
        title = ['Source dataset name:', config.source_dataset_name, 'Partition name', 'Train', 'Date and Time:', current_date_time_string]
        writer.writerow(title) # The writer writes the list of strings called "title"
        writer.writerow([]) # The writer writes a blank new line
        writer.writerow(['List of train dataset Input and GroundTruth image filenames:'])

    # Update the metadata of train dataset Input images to the CSV file
    with open(csv_result_train_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of train dataset Input images
                    writer = csv.writer(csvfile, delimiter=' ') 
                    train_Input_FileNames_List = os.listdir(train_Input_folder_path)
                    writer.writerows(train_Input_FileNames_List)
                    writer.writerow([])
                    writer = csv.writer(csvfile, delimiter=',') 
                    writer.writerow(['-------------------------Partitioning dataset operations status-------------------------'])
                    writer.writerow(['Operations completed, Summary:'])
                    writer.writerow(['Original dataset Input images:', len(src_Input_FileNames), 'Original dataset GroundTruth images:', len(src_GroundTruth_FileNames)])
                    writer.writerow(['Train dataset Input images:', len(train_Input_FileNames), 'Train dataset GroundTruth images:', len(train_GroundTruth_FileNames)])
                    writer.writerow(['Train dataset:Validation dataset',f'{train_ratio}:{val_ratio}'])
                    if check_lists(train_Input_FileNames, train_GroundTruth_FileNames): # check if train dataset Input folder and GroundTruth folder have the same elements
                        print("The train dataset Input folder and GroundTruth folder have the same elements")
                        writer.writerow(['The train dataset Input folder and GroundTruth folder have the same elements'])
                    else:
                        print("The train dataset Input folder and GroundTruth folder do not have the same elements")
                        writer.writerow(['The train dataset Input folder and GroundTruth folder do not have the same elements'])

    # CSV file to record metadata of validation dataset
    current_date_time_string = time.strftime("%Y_%m_%d-%H_%M_%S") # Get the current date and time as a string, according to the specified format (YYYY_MM_DD-HH_MM_SS)
    csv_result_validation_filename = 'Metadata-Reorganized-' + config.source_dataset_name + '-' + 'Validation' + '-' + current_date_time_string + '.csv' # Create the filename of the csv that stores the metrics data
    csv_result_validation_filepath = os.path.join(config.root_dir,csv_result_validation_filename) # Create the path to the csv that stores the metrics data
    with open(csv_result_validation_filepath, 'w', newline='') as csvfile: # Create and open an empty csv file at the path of csv_result_filepath with write mode, later we append different dictionaries of metric data as required. Now the opened csv file is called csvfile object.
        writer = csv.writer(csvfile, delimiter=',') # Create a csv writer. From the list of strings called title, the delimiter split the string after a "," into an element
        title = ['Source dataset name:', config.source_dataset_name, 'Partition name', 'Validation', 'Date and Time:', current_date_time_string]
        writer.writerow(title) # The writer writes the list of strings called "title"
        writer.writerow([]) # The writer writes a blank new line
        writer.writerow(['List of validation dataset Input and GroundTruth image filenames:'])

    # Update the metadata of validation dataset Input images to the CSV file
    with open(csv_result_validation_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_result_filepath with append mode, so that we can append the data of validation dataset Input images
                    writer = csv.writer(csvfile, delimiter=' ') 
                    validation_Input_FileNames_List = os.listdir(val_Input_folder_path)
                    writer.writerows(validation_Input_FileNames_List)
                    writer.writerow([])
                    writer = csv.writer(csvfile, delimiter=',') 
                    writer.writerow(['-------------------------Partitioning dataset operations status-------------------------'])
                    writer.writerow(['Operations completed, Summary:'])
                    writer.writerow(['Original dataset Input images:', len(src_Input_FileNames), 'Original dataset GroundTruth images:', len(src_GroundTruth_FileNames)])
                    writer.writerow(['Validation dataset Input images:', len(val_Input_FileNames), 'Validation dataset GroundTruth images:', len(val_GroundTruth_FileNames)])
                    writer.writerow(['Train dataset:Validation dataset',f'{train_ratio}:{val_ratio}'])
                    if check_lists(val_Input_FileNames, val_GroundTruth_FileNames): # check if validation dataset Input folder and GroundTruth folder have the same elements
                        print("The validation dataset Input folder and GroundTruth folder have the same elements")
                        writer.writerow(['The validation dataset Input folder and GroundTruth folder have the same elements'])
                    else:
                        print("The validation dataset Input folder and GroundTruth folder do not have the same elements")
                        writer.writerow(['The validation dataset Input folder and GroundTruth folder do not have the same elements'])

 

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser() # The parser is the ArgumentParser object that holds all the information necessary to read the command-line arguments.
    # Input Parameters
    parser.add_argument('--root_dir', type=str, default= "D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/Reorganized_sampling_SICE_Part1_TrainingSet") # The absolute path of the source dataset
    parser.add_argument('--Input_folder_basename', type=str, default= "/Input") # Base name for the Input folder for the train and validation datasets
    parser.add_argument('--GroundTruth_folder_basename', type=str, default= "/GroundTruth") # Base name for the GroundTruth folder for the train and validation datasets
    parser.add_argument('--source_dataset_name', type=str, default="SICE_Dataset_Part1") # The name of the source dataset
    # parser.add_argument('--lr', type=float, default=0.0001) # Add an argument type (optional argument) named lr. The value given to this argument type must be float data type. If no value is given to this argument type, then the default value will become the value of this argument type.
    # parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--random_seed', type=int, default=42) # The seed value used to initialize the numpy random number generator for reproducible random number generations
    parser.add_argument('--val_ratio', type=float, default=0.1) # The ratio of validation data from the original dataset
    
    config = parser.parse_args() 

    partitioning(config)