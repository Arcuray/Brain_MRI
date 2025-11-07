import torch.nn as nn
from torchvision import models
"""restnet"""
def get_model(num_classes=4, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
