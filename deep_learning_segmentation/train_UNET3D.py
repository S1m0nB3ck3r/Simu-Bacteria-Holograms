"""
Training script for 3D U-Net hologram segmentation

Usage:
    python train_UNET3D.py config_train.json
    python train_UNET3D.py path/to/custom_config.json
    
Configuration:
    All parameters are specified in a JSON configuration file
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import UNet3D, HologramToSegmentationDataset, VolumeBatchSampler, dice_coefficient


class SegmentationLoss(nn.Module):
    """Combined BCE and Dice loss for highly imbalanced segmentation
    
    Args:
        bce_weight: Weight for BCE loss component (default: 0.3)
        dice_weight: Weight for Dice loss component (default: 0.7)
        pos_weight: Weight multiplier for positive class in BCE (default: 10.0)
    """
    def __init__(self, bce_weight=0.3, dice_weight=0.7, pos_weight=10.0):
        super(SegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def dice_loss(self, pred, target):
        """Dice loss component"""
        pred = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        """Compute combined loss"""
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def train(model, dataloader, optimizer, criterion, device, verbose_timing=False):
    """Training step for one epoch"""
    model.train()
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    f1_metric = BinaryF1Score().to(device)

    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    
    # Timing measurements
    epoch_start = time.time()
    batch_start = time.time()

    for i, (input_volume, target) in enumerate(dataloader):
        data_load_time = time.time() - batch_start
        
        input_volume, target = input_volume.to(device), target.to(device)

        # Forward pass
        forward_start = time.time()
        out = model(input_volume)
        if out.shape != target.shape:
            out = torch.nn.functional.interpolate(
                out, size=target.shape[2:], mode='trilinear', align_corners=False
            )
        loss = criterion(out, target).mean()
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        backward_time = time.time() - backward_start

        preds = torch.sigmoid(out)
        precision_metric.update(preds, target.int())
        recall_metric.update(preds, target.int())
        f1_metric.update(preds, target.int())

        dice = dice_coefficient(out, target)
        total_loss += loss.item()
        total_dice += dice
        n_batches += 1

        # Print timing for each batch if verbose
        if verbose_timing:
            batch_total_time = time.time() - batch_start
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()
            print(f"  Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} | "
                  f"Dice: {dice:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | "
                  f"Total: {batch_total_time:.3f}s "
                  f"(Data: {data_load_time:.3f}s | Fwd: {forward_time:.3f}s | Bwd: {backward_time:.3f}s)")
        if verbose_timing:
            batch_total = time.time() - batch_start
            print(f"  Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Dice: {dice:.4f} | "
                  f"Total: {batch_total:.3f}s (Data: {data_load_time:.3f}s | Fwd: {forward_time:.3f}s | Bwd: {backward_time:.3f}s)")
        elif (i + 1) % max(1, len(dataloader) // 10) == 0:
            # Print progress every 10% if not verbose
            elapsed = time.time() - epoch_start
            avg_batch_time = elapsed / (i + 1)
            eta_seconds = avg_batch_time * (len(dataloader) - i - 1)
            eta_mins = eta_seconds / 60
            print(f"  Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Dice: {dice:.4f} | ETA: {eta_mins:.1f}min")
        
        batch_start = time.time()

    epoch_elapsed = time.time() - epoch_start
    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    
    if not verbose_timing:
        print(f"\n  === EPOCH TIMING ====")
        print(f"  Total epoch time: {epoch_elapsed/60:.2f} minutes ({epoch_elapsed:.1f} seconds)")
        print(f"  Avg time per batch: {epoch_elapsed/n_batches:.3f}s")
    
    print(f"\n  === METRICS ===")
    print(f"  Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | "
          f"Precision: {precision_metric.compute():.4f} | Recall: {recall_metric.compute():.4f} | "
          f"F1: {f1_metric.compute():.4f}")
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device):
    """Validation step"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    
    val_start = time.time()

    with torch.no_grad():
        for i, (input_volume, target) in enumerate(dataloader):
            input_volume, target = input_volume.to(device), target.to(device)
            
            out = model(input_volume)
            if out.shape != target.shape:
                out = torch.nn.functional.interpolate(
                    out, size=target.shape[2:], mode='trilinear', align_corners=False
                )
            loss = criterion(out, target).mean()
            dice = dice_coefficient(out, target)
            total_loss += loss.item()
            total_dice += dice
            n_batches += 1

    val_elapsed = time.time() - val_start
    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    print(f"\n  VALIDATION: Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | Time: {val_elapsed:.1f}s")
    
    return avg_loss, avg_dice


def save_model(model, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Model saved to {path}")


def plot_training_history(history, save_path=None):
    """Plot and optionally save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_dice'], label='Train Dice')
    ax2.plot(history['val_dice'], label='Validation Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
    
    plt.show()


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {config_path}")
    return config


def save_config(config, output_path):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {output_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train 3D U-Net for hologram segmentation')
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Load configuration
    print("\n" + "="*80)
    print("LOADING CONFIGURATION")
    print("="*80)
    config = load_config(args.config)
    
    # Display configuration
    print(f"\nExperiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Description: {config.get('description', 'N/A')}")
    
    # ==================== SETUP ====================
    # Device configuration
    use_cuda = config.get('device', {}).get('use_cuda_if_available', True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"\nUsing device: {device}")

    # Paths
    paths = config['paths']
    base_path = paths['data_base_path']
    model_output_dir = paths['model_output_dir']
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'unet3d')
    experiment_dir = os.path.join(model_output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Output paths
    checkpoint_prefix = config.get('output', {}).get('checkpoint_prefix', 'unet3d')
    model_path = os.path.join(experiment_dir, f"{checkpoint_prefix}_best.pth")
    history_plot_path = os.path.join(experiment_dir, "training_history.png")
    config_save_path = os.path.join(experiment_dir, "config_used.json")
    
    # Save config to experiment directory
    save_config(config, config_save_path)
    
    # Hyperparameters
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    
    BATCH_SIZE = training_config.get('batch_size', 1)
    LEARNING_RATE = training_config.get('learning_rate', 1e-4)
    NUM_EPOCHS = training_config.get('num_epochs', 50)
    PATIENCE = training_config.get('early_stopping_patience', 10)
    DROPOUT_PROB = model_config.get('dropout_prob', 0.3)
    
    # Patch parameters
    PATCH_SIZE_XY = tuple(data_config.get('patch_size_xy', [128, 128]))
    STRIDE_XY = tuple(data_config.get('stride_xy', [64, 64]))
    PATCH_SIZE_Z = data_config.get('patch_size_z', 64)
    STRIDE_Z = data_config.get('stride_z', 32)
    HOLOGRAM_PREFIX = data_config.get('hologram_prefix', 'holo_')
    SEGMENTATION_PREFIX = data_config.get('segmentation_prefix', 'segmentation_')
    
    # Output settings
    output_config = config.get('output', {})
    VERBOSE_TIMING = output_config.get('verbose_timing', False)

    # ==================== LOAD DATA ====================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Get paths
    train_dir = os.path.join(base_path, paths.get('train_subdir', 'train'))
    val_dir = os.path.join(base_path, paths.get('val_subdir', 'val'))
    
    hologram_subdir = paths.get('hologram_subdir', 'simulated_hologram')
    segmentation_subdir = paths.get('segmentation_subdir', 'binary_volume')
    
    train_holo_dir = os.path.join(train_dir, hologram_subdir)
    train_seg_dir = os.path.join(train_dir, segmentation_subdir)
    val_holo_dir = os.path.join(val_dir, hologram_subdir)
    val_seg_dir = os.path.join(val_dir, segmentation_subdir)
    
    # Verify directories exist
    if not os.path.exists(train_holo_dir):
        raise FileNotFoundError(f"Training hologram directory not found: {train_holo_dir}")
    if not os.path.exists(train_seg_dir):
        raise FileNotFoundError(f"Training segmentation directory not found: {train_seg_dir}")
    
    # Count files
    train_holo_files = [f for f in os.listdir(train_holo_dir) if f.endswith('.npy')]
    val_holo_files = [f for f in os.listdir(val_holo_dir) if f.endswith('.npy')] if os.path.exists(val_holo_dir) else []
    
    print(f"\nData directories:")
    print(f"  Train hologram: {train_holo_dir}")
    print(f"  Train segmentation: {train_seg_dir}")
    print(f"  Val hologram: {val_holo_dir}")
    print(f"  Val segmentation: {val_seg_dir}")
    
    print(f"\nData files found:")
    print(f"  - Training: {len(train_holo_files)} files")
    print(f"  - Validation: {len(val_holo_files)} files")
    
    if len(train_holo_files) == 0:
        raise ValueError("No training files found! Please run split_data.py first.")
    
    # Load parameters from first hologram file
    first_holo = os.path.join(train_holo_dir, train_holo_files[0])
    print(f"\nLoading parameters from: {os.path.basename(first_holo)}")
    
    # Load reconstruction parameters from config
    parameters = data_config.get('reconstruction_parameters', {
        "holo_size_x": 512,
        "holo_size_y": 512,
        "holo_plane_number": 128,
        "medium_wavelength": 0.66e-6,
        "magnification_cam": 2.857,
        "pix_size_cam": 3.45e-6,
        "Z_step": 5e-6
    })
    
    print(f"Using reconstruction parameters:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")

    # ==================== CREATE DATASETS ====================
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    
    train_dataset = HologramToSegmentationDataset(
        train_holo_dir,
        train_seg_dir,
        parameters,
        patch_size_XY=PATCH_SIZE_XY,
        stride_XY=STRIDE_XY,
        patch_size_Z=PATCH_SIZE_Z,
        stride_Z=STRIDE_Z,
        hologram_prefix=HOLOGRAM_PREFIX,
        segmentation_prefix=SEGMENTATION_PREFIX,
        verbose_timing=VERBOSE_TIMING
    )
    
    if len(val_holo_files) > 0:
        val_dataset = HologramToSegmentationDataset(
            val_holo_dir,
            val_seg_dir,
            parameters,
            patch_size_XY=PATCH_SIZE_XY,
            stride_XY=STRIDE_XY,
            patch_size_Z=PATCH_SIZE_Z,
            stride_Z=STRIDE_Z,
            hologram_prefix=HOLOGRAM_PREFIX,
            segmentation_prefix=SEGMENTATION_PREFIX,
            verbose_timing=VERBOSE_TIMING
        )
    else:
        print("\n  Warning: No validation data found, using training data for validation")
        val_dataset = train_dataset

    # Create samplers for efficient cache usage
    train_sampler = VolumeBatchSampler(train_dataset, shuffle=True)
    val_sampler = VolumeBatchSampler(val_dataset, shuffle=False)
    
    # Create dataloaders with custom sampler (no shuffle parameter needed)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    # ==================== INITIALIZE MODEL ====================
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = UNet3D(dropout_prob=DROPOUT_PROB).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Use combined loss for imbalanced data
    loss_config = config.get('loss', {})
    bce_weight = loss_config.get('bce_weight', 0.3)
    dice_weight = loss_config.get('dice_weight', 0.7)
    pos_weight = loss_config.get('pos_weight', 10.0)
    criterion = SegmentationLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight
    ).to(device)
    
    print(f"\nLoss configuration:")
    print(f"  - Type: Combined (BCE + Dice)")
    print(f"  - BCE weight: {bce_weight}")
    print(f"  - Dice weight: {dice_weight}")
    print(f"  - Positive class weight: {pos_weight}")
    
    lr_patience = training_config.get('lr_scheduler_patience', 5)
    lr_factor = training_config.get('lr_scheduler_factor', 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=lr_patience, factor=lr_factor
    )

    print(f"\nModel parameters:")
    print(f"  - Dropout: {DROPOUT_PROB}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Early stopping patience: {PATIENCE}")
    print(f"\nExperiment directory: {experiment_dir}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # ==================== TRAINING LOOP ====================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    training_history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_dice = train(model, train_dataloader, optimizer, criterion, device, VERBOSE_TIMING)
        training_history['train_loss'].append(train_loss)
        training_history['train_dice'].append(train_dice)
        
        # Validate
        val_loss, val_dice = validate(model, val_dataloader, criterion, device)
        training_history['val_loss'].append(val_loss)
        training_history['val_dice'].append(val_dice)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            print(f"  ✓ Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f})")
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, model_path)
            
            # Save checkpoint with optimizer state
            checkpoint_path = os.path.join(experiment_dir, f"{checkpoint_prefix}_checkpoint.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'training_history': training_history
            }, checkpoint_path)
        else:
            patience_counter += 1
            print(f"  No improvement in validation loss ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping triggered after {epoch+1} epochs")
                break

    # ==================== SAVE RESULTS ====================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Plot training history
    if config.get('output', {}).get('save_training_history_plot', True):
        plot_training_history(training_history, save_path=history_plot_path)
    else:
        plot_training_history(training_history)
    
    # Save training history as JSON
    history_json_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_json_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"Training history saved to: {history_json_path}")
    
    print("\nTraining complete!")
