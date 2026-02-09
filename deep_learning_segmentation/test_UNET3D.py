import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs'))

from simu_hologram import *
import propagation as propag
import CCL3D
import focus
import cupy as cp
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score


class VolumeReconstructor:
    def __init__(self, parameters):
        self.parameters = parameters.copy()
        self.nb_pix_X = parameters["holo_size_x"]
        self.nb_pix_Y = parameters["holo_size_y"]
        self.nb_plan = parameters["holo_plane_number"]
        self.lambda_milieu = parameters["medium_wavelength"]
        self.magnification = parameters["magnification_cam"]
        self.pix_size = parameters["pix_size_cam"]
        self.dz = parameters["Z_step"]

        self.d_fft_holo = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y), dtype=cp.complex64)
        self.d_fft_holo_propag = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y), dtype=cp.complex64)
        self.d_KERNEL = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y), dtype=cp.complex64)
        self.d_volume_module = cp.zeros(shape=(self.nb_pix_X, self.nb_pix_Y, self.nb_plan), dtype=cp.float32)

    def volume_reconstruction(self, h_holo, parameters=None):
        if parameters is not None and parameters != self.parameters:
            self.__init__(parameters)
        d_holo = cp.asarray(h_holo)
        propag.volume_propag_angular_spectrum_to_module(
            d_holo, self.d_fft_holo, self.d_KERNEL, self.d_fft_holo_propag, self.d_volume_module,
            self.lambda_milieu, self.magnification, self.pix_size, self.nb_pix_X, self.nb_pix_Y,
            0.0, self.dz, self.nb_plan, 0, 0)
        volume_np = cp.asnumpy(self.d_volume_module)
        # Normalisation pour l'entraînement
        volume_np = (volume_np - volume_np.mean()) / (volume_np.std() + 1e-8)
        volume_tensor = torch.from_numpy(volume_np).unsqueeze(0)
        return volume_tensor

class HologramToSegmentationDataset(Dataset):
    """Dataset with pre-loaded volumes for faster training"""
    def __init__(
        self,
        holo_data_files,
        patch_size_XY=(128, 128),
        stride_XY=(64, 64),
        patch_size_Z=64,
        stride_Z=32,
    ):
        self.holo_data_files = holo_data_files
        self.patch_size_XY = patch_size_XY
        self.stride_XY = stride_XY
        self.patch_size_Z = patch_size_Z
        self.stride_Z = stride_Z

        _, _, parameters, _ = load_holo_data(holo_data_files[0])
        self.reconstructor = VolumeReconstructor(parameters)
        self.slice_index = []
        
        # Pre-load all volumes and their reconstructions
        print("Pre-loading volumes... (this may take some time)")
        self.preloaded_volumes = {}
        self.preloaded_bools = {}
        
        for file_idx, path in enumerate(holo_data_files):
            print(f"  - Loading file {file_idx + 1}/{len(holo_data_files)}: {os.path.basename(path)}")
            bool_volume, hologram_image, parameters, bacteria_list = load_holo_data(path)
            
            # Reconstruct volume
            volume_tensor = self.reconstructor.volume_reconstruction(hologram_image, parameters)
            bool_tensor = torch.from_numpy(bool_volume).unsqueeze(0).to(torch.float32)
            
            # Store in dictionary
            self.preloaded_volumes[file_idx] = volume_tensor
            self.preloaded_bools[file_idx] = bool_tensor
            
            W, H, D = bool_volume.shape
            
            # Index all possible patches
            for y in range(0, H - patch_size_XY[1] + 1, stride_XY[1]):
                for x in range(0, W - patch_size_XY[0] + 1, stride_XY[0]):
                    for z in range(0, D - patch_size_Z + 1, stride_Z):
                        self.slice_index.append((file_idx, x, y, z))
        
        print(f"Pre-loading complete. Total patches: {len(self.slice_index)}")

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        file_idx, x, y, z = self.slice_index[idx]
        
        # Load from pre-loaded data
        volume_tensor = self.preloaded_volumes[file_idx]
        bool_tensor = self.preloaded_bools[file_idx]

        px, py = self.patch_size_XY
        pz = self.patch_size_Z

        # Extract patch
        volume_patch = volume_tensor[:, x:x + px, y:y + py, z:z + pz]
        bool_patch = bool_tensor[:, x:x + px, y:y + py, z:z + pz]

        return volume_patch, bool_patch

class UNet3D(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet3D, self).__init__()
        self.dropout_prob = dropout_prob
        self.encoder1 = self.block(1, 16, dropout_prob)
        self.encoder2 = self.block(16, 32, dropout_prob)
        self.encoder3 = self.block(32, 64, dropout_prob)
        self.encoder4 = self.block(64, 128, dropout_prob)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self.block(128, 256, dropout_prob)
        
        # Upsampling layers
        self.upconv4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self.block(256, 128, dropout_prob)  # 256 = 128 (upconv) + 128 (skip)
        
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self.block(128, 64, dropout_prob)  # 128 = 64 (upconv) + 64 (skip)
        
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self.block(64, 32, dropout_prob)  # 64 = 32 (upconv) + 32 (skip)
        
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = self.block(32, 16, dropout_prob)  # 32 = 16 (upconv) + 16 (skip)
        
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def block(self, in_channels, out_channels, dropout_prob=0.0):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout3d(dropout_prob))
        layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, X, Y, Z)
        # Encoder avec skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder avec skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenation avec e4
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenation avec e3
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenation avec e2
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Concatenation avec e1
        d1 = self.decoder1(d1)
        
        return self.final_conv(d1)

def dice_coefficient(preds, targets, eps=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice


def train(model, dataloader, optimizer, criterion, device):
    """Training step"""
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

    for i, (input_volume, target) in enumerate(dataloader):
        input_volume, target = input_volume.to(device), target.to(device)

        out = model(input_volume)
        if out.shape != target.shape:
            out = torch.nn.functional.interpolate(
                out, size=target.shape[2:], mode='trilinear', align_corners=False
            )
        loss = criterion(out, target).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = torch.sigmoid(out)
        precision_metric.update(preds, target.int())
        recall_metric.update(preds, target.int())
        f1_metric.update(preds, target.int())

        dice = dice_coefficient(out, target)
        total_loss += loss.item()
        total_dice += dice
        n_batches += 1

        if (i + 1) % max(1, len(dataloader) // 10) == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Dice: {dice:.4f}")

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    print(f"  Epoch Summary: Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, "
          f"Precision: {precision_metric.compute():.4f}, Recall: {recall_metric.compute():.4f}, "
          f"F1: {f1_metric.compute():.4f}")
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device):
    """Validation step"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

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

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    print(f"  Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_dice


def save_model(model, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Model saved to {path}")


def load_model(model, path, device):
    """Load model from checkpoint"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Model loaded from {path}")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure paths - MODIFY THESE ACCORDING TO YOUR DATA
    base_path = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\simu_bact_random\2025_07_03_16_46_31\data_holograms"
    model_output_dir = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\models"
    model_path = os.path.join(model_output_dir, "unet3d_best.pth")
    
    # Load data files
    train_files = [os.path.join(base_path, "train", f) for f in os.listdir(os.path.join(base_path, "train")) if f.endswith(".npz")]
    val_files = [os.path.join(base_path, "val", f) for f in os.listdir(os.path.join(base_path, "val")) if f.endswith(".npz")] if os.path.exists(os.path.join(base_path, "val")) else train_files[:len(train_files)//5]
    test_files = [os.path.join(base_path, "test", f) for f in os.listdir(os.path.join(base_path, "test")) if f.endswith(".npz")]

    print(f"\nData files found:")
    print(f"  - Training: {len(train_files)} files")
    print(f"  - Validation: {len(val_files)} files")
    print(f"  - Test: {len(test_files)} files")

    # Create datasets
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    
    train_dataset = HologramToSegmentationDataset(
        train_files,
        patch_size_XY=(128, 128),
        stride_XY=(64, 64),
        patch_size_Z=64,
        stride_Z=32
    )
    
    val_dataset = HologramToSegmentationDataset(
        val_files,
        patch_size_XY=(128, 128),
        stride_XY=(64, 64),
        patch_size_Z=64,
        stride_Z=32
    )
    
    test_dataset = HologramToSegmentationDataset(
        test_files,
        patch_size_XY=(128, 128),
        stride_XY=(64, 64),
        patch_size_Z=64,
        stride_Z=32
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = UNet3D(dropout_prob=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Training parameters
    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop with early stopping
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    training_history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_dice = train(model, train_dataloader, optimizer, criterion, device)
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
        else:
            patience_counter += 1
            print(f"  No improvement in validation loss ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)
    
    if os.path.exists(model_path):
        model = load_model(model, model_path, device)
    else:
        print("  Warning: Best model not found, using current model")

    model.eval()
    _, _, parameters, _ = load_holo_data(test_files[0])
    reconstructor = VolumeReconstructor(parameters)

    # Inference on full volumes
    num_inference = min(5, len(test_files))
    print(f"\nRunning inference on {num_inference} test samples...\n")
    
    for i in range(num_inference):
        print(f"Inference on hologram {i+1}/{num_inference}")
        bool_volume_np, holo_i, parameters_i, _ = load_holo_data(test_files[i])

        volume_tensor = reconstructor.volume_reconstruction(holo_i, parameters_i).to(device)
        target_tensor = torch.from_numpy(bool_volume_np).float().unsqueeze(0).to(device)

        X, Y, Z = volume_tensor.shape[1], volume_tensor.shape[2], volume_tensor.shape[3]
        patch_x, patch_y = 128, 128
        stride_x, stride_y = 64, 64
        patch_z = 64
        stride_z = 32

        pred_accum = torch.zeros((1, X, Y, Z), device=device)
        weight_accum = torch.zeros_like(pred_accum)

        with torch.no_grad():
            # Gestion des bords : couvrir tout le volume
            for y in range(0, Y, stride_y):
                y_end = min(y + patch_y, Y)
                y_start = max(0, y_end - patch_y)
                
                for x in range(0, X, stride_x):
                    x_end = min(x + patch_x, X)
                    x_start = max(0, x_end - patch_x)
                    
                    for z in range(0, Z, stride_z):
                        z_end = min(z + patch_z, Z)
                        z_start = max(0, z_end - patch_z)
                        
                        patch = volume_tensor[:, x_start:x_end, y_start:y_end, z_start:z_end].unsqueeze(0)
                        out_patch = model(patch)
                        if out_patch.shape != patch.shape:
                            out_patch = torch.nn.functional.interpolate(
                                out_patch, size=patch.shape[2:], mode='trilinear', align_corners=False
                            )
                        prob_patch = torch.sigmoid(out_patch)[0, 0]
                        pred_accum[:, x_start:x_end, y_start:y_end, z_start:z_end] += prob_patch
                        weight_accum[:, x_start:x_end, y_start:y_end, z_start:z_end] += 1.0

            pred_probs = pred_accum / (weight_accum + 1e-6)
            pred_mask = (pred_probs > 0.5).float()

        dice = dice_coefficient(pred_probs.unsqueeze(1), target_tensor.unsqueeze(1)).item()
        print(f"  Dice score: {dice:.4f}\n")

        # Visualization
        gt_sum = target_tensor[0].cpu().sum(dim=-1)
        pred_sum = pred_probs[0].cpu().sum(dim=-1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(gt_sum, cmap='gray')
        axs[0].set_title("Ground Truth (sum over Z)")
        axs[1].imshow(pred_sum, cmap='gray')
        axs[1].set_title("Prediction (sum over Z)")
        plt.suptitle(f"Sample {i+1} - Dice: {dice:.4f}")
        plt.tight_layout()
        plt.show()

    # Plot training history
    print("\n" + "="*80)
    print("TRAINING HISTORY")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(training_history['train_loss'], label='Train Loss')
    ax1.plot(training_history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(training_history['train_dice'], label='Train Dice')
    ax2.plot(training_history['val_dice'], label='Validation Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    print("\nTraining complete!")