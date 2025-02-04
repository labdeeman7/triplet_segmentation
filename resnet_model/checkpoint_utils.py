import torch
import json
import os
import wandb
import types

def save_checkpoint(model, optimizer, epoch, best_val_accuracy, file_path):
    """Saves a checkpoint of the model."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_accuracy': best_val_accuracy
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}", flush=True)


def load_checkpoint(config, model, optimizer=None):
    """Loads a checkpoint and restores the model and optimizer states."""
    file_path = config.load_from_checkpoint
    assert os.path.isfile(file_path), f'file path not found {file_path}' 
    
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_accuracy = checkpoint['best_val_accuracy']
    print(f"Checkpoint loaded from {file_path}, starting from epoch {epoch+1}", flush=True)
    return epoch, best_val_accuracy


def load_checkpoint_from_latest(config, model, optimizer=None, device = 'cuda'):
    """Loads a checkpoint and restores the model and optimizer states."""
    
    work_dir = config.work_dir      
    assert os.path.isdir(work_dir), f'work_dir not found {work_dir}' 
    
    checkpoint_files = [f for f in os.listdir(work_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {work_dir}")
    
    # Extract epoch numbers by splitting the filename
    epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in checkpoint_files]
    # Get the highest epoch number
    latest_epoch = max(epoch_numbers)
    latest_checkpoint = os.path.join(work_dir, f"epoch_{latest_epoch}.pth")    
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # load optimizer in cuda
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in optimizer.state.values():
            if isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
    epoch = checkpoint['epoch']
    best_val_accuracy = checkpoint['best_val_accuracy']
    print(f"Checkpoint loaded from {latest_checkpoint}, starting from epoch {epoch}", flush=True)
    return epoch, best_val_accuracy
