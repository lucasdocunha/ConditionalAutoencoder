#!/usr/bin/env python3
"""
Basic fully-connected autoencoder for MNIST (PyTorch).
Run: python3 autoencoder_basic.py
Requirements: torch, torchvision
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train(model, dataloader, epochs=5, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{epochs} - loss: {avg:.6f}")

    return model


def main():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    model = Autoencoder(latent_dim=32)
    trained = train(model, loader, epochs=5, lr=1e-3)

    torch.save(trained.state_dict(), "autoencoder_mnist.pth")
    print("Modelo salvo como autoencoder_mnist.pth")


if __name__ == "__main__":
    main()
