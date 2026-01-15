import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd 
import numpy as np 
import cv2 

import torch.nn as nn
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=1849):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x)            # (B, 8, 64, 64)

        x = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)

        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, 1849)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=1849):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 32 * 32 * 16)

        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 3, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       # (B, 16384)
        x = x.view(-1, 16, 32, 32)           # (B, 16, 32, 32)

        x = F.relu(self.conv1(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)

        x = F.relu(self.conv2(x))            # (B, 8, 64, 64)
        x = self.upsample(x)                 # (B, 8, 128, 128)

        x = self.conv3(x)                    # (B, 3, 128, 128)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=1849):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

