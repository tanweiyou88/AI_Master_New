# This script aims to delete a directory and all its contents. Run the lines below only when needed.
import shutil
dst_dir = "D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/Reorganized_sampling_SICE_Part1_TrainingSet" # Absolute path of the destination root folder 
shutil.rmtree(dst_dir) 


# dst_dir2 = "D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/Reorganized_sampling_SICE_Part1_TrainingSet/train" # Absolute path of the train folder in the destination root folder 
# dst_dir3 = "D:/AI_Master_New/Folders_for_trial_and_error/Reorganize_dataset/Reorganized_sampling_SICE_Part1_TrainingSet/val" # Absolute path of the validation folder in the destination root folder 
# shutil.rmtree(dst_dir2) 
# shutil.rmtree(dst_dir3) 