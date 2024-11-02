import torch
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.hub.set_dir("D:/AI_Master_New/pretrained_model_parameters_torchvision") # Set the path to a local folder which used to save downloaded models & parameters (weights and biases)
vit_b_16 = torchvision.models.vit_b_16(pretrained=True).to(device) # Load the pretrained model (Here is ViT that using image patch size of 16x16 pixels) from torchvision for usage, at the same time download its pretrained parameters (weights and biases) from torchvision using "pretrained=True". Move the model to cuda.
