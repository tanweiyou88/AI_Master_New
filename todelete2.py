# # import torch
# # x = torch.randn([3, 4], requires_grad=True)
# # print(x.requires_grad)
# # with torch.no_grad():
# #     y = x * 2
# #     print(x.requires_grad)
# #     print(y.requires_grad)
# # z = x * 2
# # print(z.requires_grad)





# # import necessary libraries 
# import torch 
  
# # define a tensor 
# A = torch.tensor(1., requires_grad=True) 
# print("Tensor-A:", A) 
  
# # define a function using A tensor  
# # inside loop 
# with torch.no_grad(): 
#     B = A + 1
#     print("Tensor-A:", A) 
#     print("B.requires_grad=", B.requires_grad) 
# print("Tensor-A:", A) 
# print("B.requires_grad=", B.requires_grad) 
# print("B:-", B) 
  
# # check gradient 
# print("B.requires_grad=", B.requires_grad) 

# from matplotlib import pyplot as plt

# f = plt.figure()
# print(f.get_dpi())
# print(f.get_figwidth(), f.get_figheight())

import torch
# x = torch.tensor([float('nan'), float('inf'), , 3.14])
# a = torch.tensor[-float('inf')]
# x = torch.nan_to_num(a)
# print(-torch.inf)
# print(type(-torch.inf))
# print(float('inf'))

# x = torch.tensor([1])
# x = x.cpu()

# y = torch.tensor([2])
# y = y.cuda()

# print(x)
# print(y)

# list = []
# print(list)
# list.append(['epoch'; 'epoch_average_psnr'; 'epoch_average_ssim'; 'epoch_average_mae'; 'epoch_average_lpips'; 'epoch_accumulate_number_of_val_input_samples_processed'; 'epoch_accumulate_psnr'; 'epoch_accumulate_ssim'; 'epoch_accumulate_mae';'epoch_accumulate_lpips'])
# print(list)
# list.append([1;0.02;0.02;634.02;2350.02;234.02; 0.56562; 9.02; 0.882;0.02545])
# print(list)
# print('row:', len(list))
# print('column:', len(list[0]))


# import csv
# with open('todelete.csv' , 'w', newline="") as f:
#      csvwriter = csv.writer(f , delimiter = ',')
#     #  csvwriter.writerow(fieldnames2)
#      csvwriter.writerows(list)


import pandas as pd
import numpy as np
df = pd.read_csv("D:/AI_Master_New/2025_02_05-19_05_51-Zero-DCE-dataset1-LLIE-ValidationResults-History.csv")
print(df)

# Slice a specific column, for example, 'column_name'
df_epoch_average_psnr = df['epoch_average_psnr'][:-2].values.tolist() # slice a column from the dataframe, then convert it into a list. [:-2] is to exclude the data of computational_complexity metrics results
print(df_epoch_average_psnr)
df_epoch_average_psnr = [float(i) for i in df_epoch_average_psnr] # convert each string floating point number in a list into a floating point number.
df_epoch = df['epoch'][:-2].values.tolist() 
df_epoch = [float(i) for i in df_epoch]

# Display the sliced column
print(df_epoch_average_psnr)
print(df_epoch)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 10), dpi=100, constrained_layout=True)

gs = fig.add_gridspec(2, 2) # create 2x2 grid
# (Validation results) Generate and save the figure of [Average PSNR vs Epoch]
epoch_validation_average_psnr_list_ymax = max(df_epoch_average_psnr)
print(epoch_validation_average_psnr_list_ymax)
epoch_validation_average_psnr_list_xpos = df_epoch_average_psnr.index(epoch_validation_average_psnr_list_ymax)
epoch_validation_average_psnr_list_xmax = df_epoch[epoch_validation_average_psnr_list_xpos]
ax6 = fig.add_subplot(gs[0, 0])
ax6.plot(df_epoch, df_epoch_average_psnr, 'b') #row=0, col=0, 1
ax6.plot(epoch_validation_average_psnr_list_xmax, epoch_validation_average_psnr_list_ymax, 'b', marker='o', fillstyle='none') # plot the maximum point
ax6.set_ylabel('Average PSNR [dB]') # set the y-label
ax6.set_xlabel('Epoch') # set the x-label
ax6.set_xticks(np.arange(min(df_epoch), max(df_epoch)+1, 1)) # set the interval of x-axis 
ax6.set_title(f'Y-max. coord.:[{epoch_validation_average_psnr_list_xmax},{epoch_validation_average_psnr_list_ymax:.4f}]')

plt.show()





# # **Subpart 13: Record the calculated computation complexity metrics to that csv file**
# 	with open(csv_ValidationResult_filepath, 'a', newline='') as csvfile: # Open that csv file at the path of csv_ValidationResult_filepath with append mode, so that we can append the data of ComputationalComplexity_metrics_data dictionary to that csv file.
# 		writer = csv.DictWriter(csvfile, fieldnames=ComputationalComplexity_metrics_data.keys()) # The writer (csv.DictWriter) takes the csvfile object as the csv file to write and ComputationalComplexity_metrics_data.keys() as the elements=keys of the header
# 		writer.writeheader() # The writer writes the header on the csv file
# 		writer.writerow(ComputationalComplexity_metrics_data)  # The writer writes the data (value) of ComputationalComplexity_metrics_data dictionary in sequence as a row on the csv file


