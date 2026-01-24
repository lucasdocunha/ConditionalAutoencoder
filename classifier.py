import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes=2):
        super().__init__()

        self.encoder = encoder  # encoder pré-treinado
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.encoder(x)      # saída do encoder
        z = self.dropout(z)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        logits = self.fc3(z)
        return logits
    
    