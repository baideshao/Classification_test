import os
import torch

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
