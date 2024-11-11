import torch

w = torch.empty(0, 3, 256, 256)  # create an empty tensor that has no element inside, but each of its element should have the shape of (3,256,256)

print(w)
