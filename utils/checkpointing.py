# utils/checkpointing.py
import torch
import gzip
import shutil

import os

def save_checkpoint_compressed(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    with open(path, 'wb') as f:
        torch.save(checkpoint, f)
    with open(path, 'rb') as f_in, gzip.open(f"{path}.gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)

def load_checkpoint_compressed(path, model, optimizer):
    with gzip.open(path, 'rb') as f_in, open('temp.pth', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    checkpoint = torch.load('temp.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    os.remove('temp.pth')
    return epoch
