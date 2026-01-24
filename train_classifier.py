import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from classifier import Classifier
from autoencoders import *
from autoencoders_with_skip_connections import *


# =========================
# Dataset
# =========================
class CustomImageDataset(Dataset):
    def __init__(self, csv, transform=None, autoencoder=True):
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.autoencoder = autoencoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        if self.autoencoder:
            label = image

        return image, label

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# MLflow setup
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# =========================
# Training loop
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

batche_sizes_csv = [64, 128, 256, 512, 1024]
datasets_encoder = ['CNR', 'PKLot']
datasets_classifer = ["CNR", "PKLot", "PUC", "UFPR04", "UFPR05", "camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8", "camera9"]
encoders = [
    Encoder0, Encoder1, Encoder2, Encoder3, Encoder4, 
    Encoder5, Encoder6, Encoder7, Encoder8, Encoder9,
    SkipEncoder0, SkipEncoder1, SkipEncoder2, SkipEncoder3,
    SkipEncoder4, SkipEncoder5, SkipEncoder6, SkipEncoder7,
    SkipEncoder8, SkipEncoder9
]

num_epochs = 10
batch_size = 32
lr = 1e-3

for dataset_encoder in datasets_encoder:
    for model_encoder in encoders:
        
        encoder = model_encoder().to(device)
        encoder.load_state_dict(torch.load(f"models/{model_encoder.__name__}_{dataset_encoder}/encoder.pth", map_location=device))
        
        for p in encoder.parameters():
            p.requires_grad = False

        latent_dim = encoder.latent_dim        
        for dataset_name in datasets_classifer:
            for batch_size_csv in batche_sizes_csv:
                model = Classifier(encoder, latent_dim=latent_dim, num_classes=2).to(device)

                train_dataset = CustomImageDataset(
                    f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/batches/{batch_size_csv}.csv",
                    transform=transform,
                    autoencoder=False
                )
                val_dataset = CustomImageDataset(
                    f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_validation.csv",
                    transform=transform,
                    autoencoder=False
                )


                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                
                model_name = f"{model_encoder.__name__}_Classifier"
                with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):

                    mlflow.log_param("model", model_name)
                    mlflow.log_param("encoder_dataset", dataset_encoder)
                    mlflow.log_param("dataset_classifier", dataset_name)
                    mlflow.log_param("epochs", num_epochs)
                    mlflow.log_param("batch_size", batch_size_csv)
                    mlflow.log_param("lr", lr)
                    mlflow.log_param("loss", "CrossEntropy")
                    mlflow.log_param("input_shape", "3x128x128")
                    
                    model = model.to(device)
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    for epoch in range(num_epochs):
                        model.train()
                        train_loss = 0.0
                        train_correct = 0
                        train_total = 0

                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)

                            out = model(x)                 # logits
                            loss = criterion(out, y)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()

                            preds = torch.argmax(out, dim=1)
                            train_correct += (preds == y).sum().item()
                            train_total += y.size(0)

                        train_loss /= len(train_loader)
                        train_acc = train_correct / train_total

                        model.eval()
                        val_loss = 0.0
                        val_correct = 0
                        val_total = 0

                        with torch.no_grad():
                            for x, y in val_loader:
                                x, y = x.to(device), y.to(device)

                                out = model(x)             # logits
                                loss = criterion(out, y)

                                val_loss += loss.item()

                                preds = torch.argmax(out, dim=1)
                                val_correct += (preds == y).sum().item()
                                val_total += y.size(0)

                        val_loss /= len(val_loader)
                        val_acc = val_correct / val_total


                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("train_acc", train_acc, step=epoch)
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                        mlflow.log_metric("val_acc", val_acc, step=epoch)

                    for test_dataset_name in datasets_classifer:
                        
                        test_dataset = CustomImageDataset(
                            f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{test_dataset_name}/{test_dataset_name}_test.csv",
                            transform=transform,
                            autoencoder=False
                        )
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        with torch.no_grad():
                            for x, y in test_loader:
                                x, y = x.to(device), y.to(device)
                                out = model(x)

                                preds = torch.argmax(out, dim=1)
                                test_correct += (preds == y).sum().item()
                                test_total += y.size(0)

                        test_acc = test_correct / test_total
                        mlflow.log_metric(f"test_acc-{test_dataset_name}", test_acc)

                    mlflow.pytorch.log_model(model, "classifier")

                    print(f"✓ {model_name} | {dataset_name} finalizado")

                    os.makedirs(f"models/{model_name}_{dataset_name}", exist_ok=True)

                    torch.save(model.state_dict(), f"models/{model_name}_{dataset_name}/classifier-{batch_size_csv}.pth")