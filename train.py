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
from pytorch_msssim import ssim


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
def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    device = image.device

    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std  = torch.tensor(std, device=device).view(1, -1, 1, 1)

    return image * std + mean



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
    Autoencoder0, Autoencoder1, Autoencoder2,
    Autoencoder3, Autoencoder4, Autoencoder5,
    Autoencoder6, Autoencoder7, Autoencoder8,
    Autoencoder9, 
    SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
    SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
    SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
    SkipAutoencoder9
]

num_epochs = 2
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

            # -------------------------
            # Train / Val
            # -------------------------
            for epoch in range(num_epochs):

                model.train()
                train_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)

                    out = model(x)
                    loss = criterion(out, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        val_loss += criterion(model(x), y).item()

                val_loss /= len(val_loader)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            # -------------------------
            # Test + Plot
            # -------------------------
            model.eval()
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x[:8].to(device)
                    recon = model(x)
                    break

            plot_path = plot_reconstruction(
                x, recon, model_name, dataset_name
            )
            
            test_ssim = 0.0

            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    outputs = model(images)

                    images_dn = denormalize(images)
                    outputs_dn = denormalize(outputs)

                    test_ssim += ssim(
                        outputs_dn,
                        images_dn,
                        data_range=1.0,
                        size_average=True
                    ).item()

            test_ssim /= len(test_loader)

            mlflow.log_metric("test_ssim", test_ssim)

            mlflow.log_artifact(plot_path, artifact_path="reconstructions")

            # -------------------------
            # Models
            # -------------------------
            mlflow.pytorch.log_model(model, "autoencoder")

            if hasattr(model, "encoder"):
                mlflow.pytorch.log_model(model.encoder, "encoder")
            if hasattr(model, "decoder"):
                mlflow.pytorch.log_model(model.decoder, "decoder")

            print(f"✓ {model_name} | {dataset_name} finalizado")

            os.makedirs(f"models/{model_name}_{dataset_name}", exist_ok=True)

            torch.save(model.state_dict(), f"models/{model_name}_{dataset_name}/autoencoder.pth")
            torch.save(model.encoder.state_dict(), f"models/{model_name}_{dataset_name}/encoder.pth")
            torch.save(model.decoder.state_dict(), f"models/{model_name}_{dataset_name}/decoder.pth")
        