"""
References:
1. https://towardsai.net/p/l/how-to-set-up-and-run-cuda-operations-in-pytorch
2. https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu
"""

import torch

print()

# To print Cuda version
print('The CUDA version for this PyTorch:', torch.version.cuda)

# To check whether CUDA is supported
print('Whether CUDA is supported by our system:', torch.cuda.is_available())

# To print the number of GPUs available
print('The number of GPUs available:', torch.cuda.device_count())

print('\n')

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 

#Additional Info when using cuda
if device.type == 'cuda':
    print('Name of the device:', torch.cuda.get_device_name(0))
    print('Device ID:', torch.cuda.current_device())
    print('Memory Usage:')
    print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('\tCached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')