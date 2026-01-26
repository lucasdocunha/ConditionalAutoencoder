import torch
import os
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix

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

def log_confusion_matrix_mlflow(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
):
    """
    Plota e salva uma confusion matrix no MLflow (artifact),
    usando apenas matplotlib.

    Args:
        y_true (array-like): rótulos verdadeiros
        y_pred (array-like): rótulos preditos
        class_names (list[str], optional): nomes das classes
        title (str): título do gráfico
        artifact_name (str): nome do arquivo no MLflow
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    if class_names is not None:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    plt.tight_layout()

    plt.close(fig)

    return fig
