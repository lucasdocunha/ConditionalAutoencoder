import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import mlflow 

from src.config import *
from src.utils.datasets import CustomImageDataset
from src.utils.transform import return_transform
from src.models import *
from src.utils.plot import log_confusion_matrix_mlflow
from sklearn.metrics import accuracy_score, precision_score, f1_score




def train_classifier(
    gpu_id=0,
    model_encoder=None,
    dataset_encoder_name=None,
    dataset_classifier_name=None,
    batch_size=32,
    num_epochs=10,
    lr=1e-3
):
    device = Config.DEVICES[gpu_id]
    batche_sizes_csv = [64, 128, 256, 512, 1024]
    datasets_test = ["PUC", "UFPR04", "UFPR05", "camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8", "camera9"]
    transform = return_transform()
    
    mlflow.set_experiment(f"Classifier_{model_encoder.__name__[-1]}")        
          
    for batch_size_csv in batche_sizes_csv:
        encoder = model_encoder().to(device)
        encoder.load_state_dict(torch.load(f"models/Autoencoder{model_encoder.__name__[-1]}_{dataset_encoder_name}/encoder.pth", map_location=device))
        
        for p in encoder.parameters():
            p.requires_grad = False

        latent_dim = encoder.latent_dim
          
        model = Classifier(encoder, latent_dim=latent_dim, num_classes=2).to(device)
        
        train_dataset = CustomImageDataset(
            f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_classifier_name}/batches/batch-{batch_size_csv}.csv",
            transform=transform,
            autoencoder=False
        )
        val_dataset = CustomImageDataset(
            f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_classifier_name}/{dataset_classifier_name}_validation.csv",
            transform=transform,
            autoencoder=False
        )


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        
        model_name = f"{model_encoder.__name__}_Classifier"
        with mlflow.start_run(run_name=f"{model_name}_{dataset_classifier_name}"):

            mlflow.log_param("model", model_name)
            mlflow.log_param("encoder_dataset", dataset_encoder_name)
            mlflow.log_param("dataset_classifier", dataset_classifier_name)
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("n_images_to_train", batch_size_csv)
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

            for test_dataset_name in datasets_test:

                y_true = []
                y_pred = []

                test_dataset = CustomImageDataset(
                    f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{test_dataset_name}/{test_dataset_name}_test.csv",
                    transform=transform,
                    autoencoder=False
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False
                )

                model.eval()
                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        out = model(x)
                        preds = torch.argmax(out, dim=1)

                        y_true.extend(y.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())

                # métricas
                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="binary")
                f1 = f1_score(y_true, y_pred, average="binary")

                # log no MLflow
                mlflow.log_metric(f"test_acc-{test_dataset_name}", acc)
                mlflow.log_metric(f"test_precision-{test_dataset_name}", precision)
                mlflow.log_metric(f"test_f1-{test_dataset_name}", f1)

                cm = log_confusion_matrix_mlflow(
                    y_true=y_true,
                    y_pred=y_pred,
                    class_names=["empty", "occupied"],
                    title=f"Confusion Matrix - {model_name} - {test_dataset_name}",
                    artifact_name=f"confusion_matrix_{test_dataset_name}.png"
                )

                mlflow.log_figure(cm, f"confusion_matrix_{test_dataset_name}.png")

            mlflow.pytorch.log_model(model, "classifier")

            print(f"✓ {model_name} | {dataset_classifier_name} finalizado")

            os.makedirs(f"models/{model_name}_{dataset_classifier_name}", exist_ok=True)

            torch.save(model.state_dict(), f"models/{model_name}_{dataset_classifier_name}/classifier-{batch_size_csv}.pth")

def worker(rank, jobs_split):
    mlflow.set_tracking_uri(Config.IP_LOCAL)
    torch.cuda.set_device(rank)

    my_jobs = jobs_split[rank]

    print(f"[GPU {rank}] recebeu {len(my_jobs)} jobs")

    for model_encoder, dataset_encoder, dataset_classifier, epochs in my_jobs:
        print("dataset de classifier: ", dataset_classifier)
        train_classifier(
            gpu_id=rank,
            model_encoder=model_encoder,
            dataset_encoder_name=dataset_encoder,
            dataset_classifier_name=dataset_classifier,                                 
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

    encoders = [
        Encoder0, Encoder1, Encoder2,
        Encoder3, Encoder4, Encoder5,
        Encoder6, Encoder7, Encoder8,
        Encoder9,
        Encoder0, Encoder1, Encoder2,
        Encoder3, Encoder4, Encoder5,
        Encoder6, Encoder7, Encoder8,
        Encoder9
    ]

    datasets_classifier = ["PUC", "UFPR04", "UFPR05", "camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8", "camera9"]
    
    jobs = []
    n_procs = len(Config.DEVICES)

    for dataset_encoder in ["CNR", "PKLot"]:                
        for model in encoders:
            for dataset_classifier in datasets_classifier:
                jobs.append((model, dataset_encoder, dataset_classifier, epochs))

    jobs_split = [jobs[i::n_procs] for i in range(n_procs)]

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(jobs_split,),
        nprocs=n_procs,
        join=True
    )
    
    