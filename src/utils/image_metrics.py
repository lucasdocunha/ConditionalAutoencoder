import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio
from pytorch_msssim import ssim, ms_ssim
import sewar.full_ref

def calculate_mse_torch(image1, image2):
    return F.mse_loss(image1, image2)

def calculate_ssim_torch(image1, image2):
    """
    image shape: (B, C, H, W)
    valores normalizados [0,1]
    """
    return ssim(image1, image2, data_range=1.0).item()

def calculate_ssim_torch(image1, image2):
    """
    image shape: (B, C, H, W)
    valores normalizados [0,1]
    """
    return ssim(image1, image2, data_range=1.0).item()

def calculate_msssim_torch(image1, image2):
    return ms_ssim(image1, image2, data_range=1.0).item()

def calculate_psnr_torch(image1, image2):
    return peak_signal_noise_ratio(image1, image2, data_range=1.0).item()

def calculate_ncc_torch(image1, image2):
    """
    Funciona com batch ou sem batch
    """
    x = image1 - image1.mean()
    y = image2 - image2.mean()

    num = torch.sum(x * y)
    denom = torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2))

    return (num / (denom + 1e-8)).item()

def calculate_vif_torch(image1, image2):
    if torch.is_tensor(image1):
        image1 = image1.detach().cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.detach().cpu().numpy()

    return sewar.full_ref.vifp(image1, image2)

def calculate_scc_torch(image1, image2):
    if torch.is_tensor(image1):
        image1 = image1.detach().cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.detach().cpu().numpy()

    return sewar.full_ref.scc(image1, image2)

def calculate_rmse_torch(image1, image2):
    if torch.is_tensor(image1):
        image1 = image1.detach().cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.detach().cpu().numpy()

    return sewar.full_ref.rmse(image1, image2)

def calculate_all_metrics_torch(image1, image2):
    """
    image1, image2:
    - shape: (B, C, H, W)
    - normalizadas [0,1]
    """

    metrics = {}

    metrics["MSE"] = calculate_mse_torch(image1, image2).item()
    metrics["SSIM"] = calculate_ssim_torch(image1, image2)
    metrics["MSSSIM"] = calculate_msssim_torch(image1, image2)
    metrics["PSNR"] = calculate_psnr_torch(image1, image2)
    metrics["NCC"] = calculate_ncc_torch(image1, image2)
    metrics["VIF"] = calculate_vif_torch(image1, image2)
    metrics["SCC"] = calculate_scc_torch(image1, image2)
    metrics["RMSE"] = calculate_rmse_torch(image1, image2)

    return metrics
