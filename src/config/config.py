import torch

class Config:
    IP_LOCAL = "http://127.0.0.1:5000"
    DEVICE0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    DEVICES = [DEVICE0, DEVICE1]