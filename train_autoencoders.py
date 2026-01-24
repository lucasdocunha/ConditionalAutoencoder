import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import mlflow 
import mlflow.pytorch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio
import sewar.full_ref


from src.config import *
from src.utils.datasets import CustomImageDataset
from src.utils.plot import plot_reconstruction, denormalize
from src.utils.transform import return_transform
from src.models import *
from src.utils.image_metrics import calculate_all_metrics_torch

def train_experiment_autoencoder(
    gpu_id=0, 
    model_class=None, 
    dataset_name=None,
    batch_size=32,
    num_epochs=10,
    lr=1e-3
):
    device = Config.DEVICES[gpu_id]
    torch.cuda.set_device(device)

    
    transform = return_transform()  
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    
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
        mlflow.log_artifact(plot_path, artifact_path="reconstructions")


        metrics_sum = {
            "MSE": 0.0,
            "SSIM": 0.0,
            "PSNR": 0.0,
            "NCC": 0.0,
            "VIF": 0.0,
            "SCC": 0.0
        }

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):

                if i >= 2:
                    break

                images = images.to(device)
                outputs = model(images)

                images_dn = torch.clamp(denormalize(images), 0, 1)
                outputs_dn = torch.clamp(denormalize(outputs), 0, 1)

                batch_metrics = calculate_all_metrics_torch(
                    images_dn, outputs_dn
                )

                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k]
        

        metrics_avg = {
            k: float(np.mean(v)) for k, v in metrics_sum.items()
        }
        for name, value in metrics_avg.items():
            mlflow.log_metric(f"test_{name.lower()}", value)


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


def worker(rank, jobs_split):
    mlflow.set_tracking_uri(Config.IP_LOCAL)
    torch.cuda.set_device(rank)

    my_jobs = jobs_split[rank]

    print(f"[GPU {rank}] recebeu {len(my_jobs)} jobs")

    for model_class, dataset_name, epochs in my_jobs:
        train_experiment_autoencoder(
            gpu_id=rank,
            model_class=model_class,
            dataset_name=dataset_name,
            batch_size=32,
            num_epochs=epochs,
            lr=1e-3
        )
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número de épocas para treinamento")
    
    args = parser.parse_args()
    
    epochs = args.epochs
    #config = Config()

    encoders = [
        Autoencoder0, Autoencoder1, Autoencoder2,
        Autoencoder3, Autoencoder4, Autoencoder5,
        Autoencoder6, Autoencoder7, Autoencoder8,
        Autoencoder9,
        SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
        SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
        SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
        SkipAutoencoder9
    ]

    jobs = []
    n_procs = len(Config.DEVICES)

    for dataset_encoder in ["CNR", "PKLot"]:
        for model in encoders:
            jobs.append((model, dataset_encoder, epochs))

    jobs_split = [jobs[i::n_procs] for i in range(n_procs)]

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(jobs_split,),
        nprocs=n_procs,
        join=True
    )
    
    
