import torch
import os
import matplotlib.pyplot as plt

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