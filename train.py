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
import mlflow.pytorch

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
# Utils
# =========================
def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    image = image * std + mean
    return torch.clamp(image, 0, 1)


def plot_reconstruction(original, reconstructed, model_name, dataset_name, save_path="./results/"):
    os.makedirs(save_path, exist_ok=True)

    original = denormalize(original.cpu())
    reconstructed = denormalize(reconstructed.detach().cpu())

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle(f"{model_name} - {dataset_name} - Reconstruction", fontsize=16)

    for i in range(8):
        axes[0, i].imshow(original[i].permute(1, 2, 0))
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
        axes[1, i].axis("off")

    plt.tight_layout()
    filename = f"{save_path}/{model_name}_{dataset_name}_reconstruction.png"
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()
    return filename


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

datasets = ["CNR", "PKLot"]
models = [
    SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
    SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
    SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
    SkipAutoencoder9
]

num_epochs = 1
batch_size = 32
lr = 1e-3

for dataset_name in datasets:

    train_dataset = CustomImageDataset(
        f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_autoencoder_train.csv",
        transform=transform
    )
    val_dataset = CustomImageDataset(
        f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_autoencoder_validation.csv",
        transform=transform
    )
    test_dataset = CustomImageDataset(
        f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_autoencoder_test.csv",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for model_class in models:

        model_name = model_class.__name__
        mlflow.set_experiment(model_name)

        with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):

            # -------------------------
            # Params
            # -------------------------
            mlflow.log_param("model", model_name)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lr", lr)
            mlflow.log_param("loss", "MSE")
            mlflow.log_param("input_shape", "3x128x128")

            model = model_class().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 1
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Treino
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images = images.to('cuda')
                labels = labels.to('cuda')
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        print(f"✓ Treinamento concluído!")
        
        # Salvar modelo
        os.makedirs('./models', exist_ok=True)
        torch.save(model.state_dict(), f'./models/{model_name}_{train}.pth')
        print(f"✓ Modelo salvo: ./models/{model_name}_{train}.pth")
        
        # Teste e Plot - Pegar primeiras 8 imagens
        model.eval()
        test_images, test_labels = None, None
        with torch.no_grad():
            for images, labels in test_loader:
                test_images = images[:8].to('cuda')
                test_labels = labels[:8].to('cuda')
                break
        
        # Predição
        with torch.no_grad():
            reconstructed = model(test_images)
        
        # Plot
        plot_reconstruction(test_images, reconstructed, model_name, train, save_path='./results/')
        
        print(f"✓ {model_name} - {train} finalizado!\n")
