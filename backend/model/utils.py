"""
Utility functions for training and inference
"""
import torch
import random
import numpy as np
from pathlib import Path
import json
from typing import List


def get_dataset_class_names(data_dir: str) -> List[str]:
    """
    Inspect data_dir/train and return the sorted list of class folder names
    exactly as torchvision.datasets.ImageFolder would use them.
    
    This is the SINGLE SOURCE OF TRUTH for class ordering.
    
    Args:
        data_dir: Path to data directory (contains train/ and val/ subdirs)
    
    Returns:
        Sorted list of class names (folder names under data_dir/train)
    """
    train_dir = Path(data_dir) / 'train'
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # Get sorted list of subdirectories (exactly like ImageFolder does)
    class_names = sorted([
        d.name for d in train_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    if len(class_names) == 0:
        raise ValueError(f"No class folders found in {train_dir}")
    
    return class_names


def save_class_names_file(class_names: List[str], filepath: str) -> None:
    """
    Save class names to a text file (one per line, UTF-8).
    This should be called during training to keep class_names.txt in sync.
    
    Args:
        class_names: List of class names
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"Class names file saved: {filepath} ({len(class_names)} classes)")


def load_class_names_file(filepath: str) -> List[str]:
    """
    Load class names from a text file (one per line).
    
    Args:
        filepath: Path to the class names file
    
    Returns:
        List of class names (empty lines are filtered out)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def save_checkpoint(model, optimizer, epoch, val_acc, filepath, class_names=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device=None, expected_num_classes=None):
    """
    Load model checkpoint with optional validation.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load to (auto-detect if None)
        expected_num_classes: If provided, verify checkpoint has matching class count
    
    Returns:
        epoch, val_acc, class_names
    
    Raises:
        ValueError: If checkpoint class count doesn't match expected
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Get class_names from checkpoint
    class_names = checkpoint.get('class_names', None)
    
    # Determine number of classes from checkpoint weights
    state_dict = checkpoint['model_state_dict']
    if 'fc.weight' in state_dict:
        ckpt_num_classes = state_dict['fc.weight'].shape[0]
    elif 'classifier.1.weight' in state_dict:
        ckpt_num_classes = state_dict['classifier.1.weight'].shape[0]
    else:
        ckpt_num_classes = len(class_names) if class_names else None
    
    # Validate class count if expected_num_classes is provided
    if expected_num_classes is not None and ckpt_num_classes is not None:
        if ckpt_num_classes != expected_num_classes:
            raise ValueError(
                f"Checkpoint class count mismatch!\n"
                f"  Checkpoint has: {ckpt_num_classes} classes\n"
                f"  Current dataset has: {expected_num_classes} classes\n"
                f"  This checkpoint was trained with a different dataset.\n"
                f"  Please retrain the model or use a compatible checkpoint."
            )
    
    # Validate class_names length matches weights
    if class_names is not None and ckpt_num_classes is not None:
        if len(class_names) != ckpt_num_classes:
            print(
                f"Warning: Checkpoint has {ckpt_num_classes} output classes but "
                f"{len(class_names)} class_names. Trimming class_names to match."
            )
            class_names = class_names[:ckpt_num_classes]
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Val Acc: {val_acc:.2f}%, Classes: {ckpt_num_classes}")
    
    return epoch, val_acc, class_names

def save_class_mapping(class_names, filepath):
    """Save class name to index mapping."""
    mapping = {name: idx for idx, name in enumerate(class_names)}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Class mapping saved to {filepath}")

def load_class_mapping(filepath):
    """Load class name to index mapping."""
    with open(filepath, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping

