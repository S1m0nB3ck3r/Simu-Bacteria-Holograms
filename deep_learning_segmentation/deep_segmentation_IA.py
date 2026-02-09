import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs'))

from simu_hologram import *
import propagation as propag
import CCL3D
import focus
import cupy as cp
import matplotlib.pyplot as plt
import wandb
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

def initialize_wandb(api_path, projet, entity):
    if os.path.exists(api_path):
        with open(api_path, 'r') as file:
            api_key = file.read().strip()
        if api_key:
            try:
                wandb.login(key=api_key)
                wandb.init(project=projet, entity=entity)
                print("Weights & Biases initialized successfully.")
                return True
            except Exception as e:
                print(f"Failed to initialize Weights & Biases: {e}")
                return False
        else:
            print("No API key found in the file.")
            return False
    else:
        print("W&B API key file not found.")
        return False

class VolumeReconstructor:
    def __init__(self, parameters):
        self.parameters = parameters.copy()
        self.nb_pix_X = parameters["holo_size_x"]
        self.nb_pix_Y = parameters["holo_size_y"]
        self.nb_plan = parameters["holo_plane_number"]
        self.lambdaMilieu = parameters["medium_wavelength"]
        self.magnification = parameters["magnification_cam"]
        self.pixSize = parameters["pix_size_cam"]
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
            self.lambdaMilieu, self.magnification, self.pixSize, self.nb_pix_X, self.nb_pix_Y,
            0.0, self.dz, self.nb_plan, 0, 0)
        volume_tensor = torch.from_numpy(cp.asnumpy(self.d_volume_module)).unsqueeze(0)
        return volume_tensor

class HologramToSegmentationDataset(Dataset):
    def __init__(self, holo_data_files, patch_size_XY=(128, 128), stride_XY=(64, 64), patch_size_Z=64, stride_Z=32):
        self.holo_data_files = holo_data_files
        self.patch_size_XY = patch_size_XY
        self.stride_XY = stride_XY
        self.patch_size_Z = patch_size_Z
        self.stride_Z = stride_Z
        _, _, parameters, _ = load_holo_data(holo_data_files[0])
        self.reconstructor = VolumeReconstructor(parameters)
        self.slice_index = []
        for file_idx, path in enumerate(holo_data_files):
            bool_volume, _, _, _ = load_holo_data(path)
            W, H, D = bool_volume.shape
            for y in range(0, H - patch_size_XY[1] + 1, stride_XY[1]):
                for x in range(0, W - patch_size_XY[0] + 1, stride_XY[0]):
                    for z in range(0, D - patch_size_Z + 1, stride_Z):
                        self.slice_index.append((file_idx, x, y, z))

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        file_idx, x, y, z = self.slice_index[idx]
        hologram_volume, hologram_image, parameters, bacteria_list = load_holo_data(self.holo_data_files[file_idx])
        volume_tensor = self.reconstructor.volume_reconstruction(hologram_image, parameters)
        bool_tensor = torch.from_numpy(hologram_volume).unsqueeze(0).to(torch.float32)
        px, py = self.patch_size_XY
        pz = self.patch_size_Z
        volume_patch = volume_tensor[:, x:x + px, y:y + py, z:z + pz]
        bool_patch = bool_tensor[:, x:x + px, y:y + py, z:z + pz]
        return parameters, bacteria_list, volume_patch, bool_patch

class UNet3D(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(UNet3D, self).__init__()
        self.dropout_prob = dropout_prob
        self.encoder1 = self.block(1, 16)
        self.encoder2 = self.block(16, 32)
        self.encoder3 = self.block(32, 64)
        self.encoder4 = self.block(64, 128)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self.block(128, 256)
        self.up4 = self.up_block(256, 128)
        self.up3 = self.up_block(128, 64)
        self.up2 = self.up_block(64, 32)
        self.up1 = self.up_block(32, 16)
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def block(self, in_channels, out_channels):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if self.dropout_prob > 0:
            layers.append(nn.Dropout3d(self.dropout_prob))
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            self.block(out_channels, out_channels)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d3 = self.up3(d4 + e4)
        d2 = self.up2(d3 + e3)
        d1 = self.up1(d2 + e2)
        return self.final_conv(d1 + e1)

def dice_coefficient(preds, targets, eps=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice

def train(model, dataloader, optimizer, criterion, device):
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

    for i, (params, bacteria_list, input_volume, target) in enumerate(dataloader):
        print(f"Processing batch {i}")
        input_volume, target = input_volume.to(device), target.to(device)

        out = model(input_volume)
        if out.shape != target.shape:
            out = torch.nn.functional.interpolate(out, size=target.shape[2:], mode='trilinear', align_corners=False)
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
        print(f"Batch {i}, Loss: {loss.item():.4f}, Dice: {dice:.4f}")

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    print(f"Epoch Summary: Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, "
          f"Precision: {precision_metric.compute():.4f}, Recall: {recall_metric.compute():.4f}, "
          f"F1: {f1_metric.compute():.4f}")
    return avg_loss, avg_dice

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, (params, bacteria_list, input_volume, target) in enumerate(dataloader):
            input_volume, target = input_volume.to(device), target.to(device)
            out = model(input_volume)
            if out.shape != target.shape:
                out = torch.nn.functional.interpolate(out, size=target.shape[2:], mode='trilinear', align_corners=False)
            loss = criterion(out, target).mean()
            dice = dice_coefficient(out, target)
            total_loss += loss.item()
            total_dice += dice
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    print(f"Validation Summary: Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\simu_bact_random\2025_07_03_16_46_31\data_holograms"
    train_files = [os.path.join(base_path, "train", f) for f in os.listdir(os.path.join(base_path, "train")) if f.endswith(".npz")]
    test_files = [os.path.join(base_path, "test", f) for f in os.listdir(os.path.join(base_path, "test")) if f.endswith(".npz")]
    model_path = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\simu_bact_random\model_unet3d.pth"

    # Load parameters from the first training file
    _, _, parameters, _ = load_holo_data(train_files[0])

    # Initialize Weights & Biases
    api_key_path = os.path.join(os.getcwd(), "wandb_api.txt")
    # wandb_initialized = initialize_wandb(api_key_path, projet="Bacteria 3D Segmentation", entity="university_of_lorraine")
    wandb_initialized = False  # Set to False for testing without W&B

    # Create datasets and dataloaders
    train_dataset = HologramToSegmentationDataset(train_files)
    test_dataset = HologramToSegmentationDataset(test_files)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Initialize model, optimizer, and criterion
    model = UNet3D(dropout_prob=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Training loop
    load_existing = True
    use_amp = True  # Enable mixed precision training

    if load_existing and os.path.exists(model_path):
        load_model(model, model_path, device)
    else:
        for epoch in range(10):
            print(f"Epoch {epoch+1}/10")
            train_loss, train_dice = train(model, train_dataloader, optimizer, criterion, device)
            val_loss, val_dice = validate(model, test_dataloader, criterion, device)

            # Log metrics to Weights & Biases if initialized
            if wandb_initialized:
                wandb.log({"Train Loss": train_loss, "Train Dice": train_dice,
                           "Validation Loss": val_loss, "Validation Dice": val_dice})

            # Step the scheduler
            scheduler.step(val_loss)

        save_model(model, model_path)

    # Close the Weights & Biases run if initialized
    if wandb_initialized:
        wandb.finish()

    # Evaluation on test data
    model.eval()
    reconstructor = VolumeReconstructor(parameters)
    for i in range(min(5, len(test_files))):
        print(f"\nInference on hologram {i}")
        bool_volume_np, holo_i, parameters_i, _ = load_holo_data(test_files[i])
        volume_tensor = reconstructor.volume_reconstruction(holo_i, parameters_i).to(device)
        target_tensor = torch.from_numpy(bool_volume_np).float().unsqueeze(0).to(device)
        X, Y, Z = volume_tensor.shape[1:]
        patch_x, patch_y = 256, 256
        stride_x, stride_y = 256, 256
        patch_z = 128
        stride_z = 128
        pred_accum = torch.zeros((1, X, Y, Z), device=device)
        weight_accum = torch.zeros_like(pred_accum)

        nb_x = (X - patch_x) // stride_x + 1
        nb_y = (Y - patch_y) // stride_y + 1
        nb_z = (Z - patch_z) // stride_z + 1

        i = j = k = 0

        with torch.inference_mode():
            for y in range(0, Y - patch_y + 1, stride_y):
                j+=1
                i = 0
                k = 0
                for x in range(0, X - patch_x + 1, stride_x):
                    i+=1
                    k = 0
                    for z in range(0, Z - patch_z + 1, stride_z):
                        k+=1    
                        print(f"Processing patch at ({i}, {j}, {k}) sur ({nb_x}, {nb_y}, {nb_z})")
                        patch = volume_tensor[:, x:x+patch_x, y:y+patch_y, z:z+patch_z].unsqueeze(1)
                        out_patch = model(patch)
                        if out_patch.shape != patch.shape:
                            out_patch = torch.nn.functional.interpolate(out_patch, size=patch.shape[2:], mode='trilinear', align_corners=False)
                        prob_patch = torch.sigmoid(out_patch)[:, 0]
                        pred_accum[:, x:x+patch_x, y:y+patch_y, z:z+patch_z] += prob_patch
                        weight_accum[:, x:x+patch_x, y:y+patch_y, z:z+patch_z] += 1.0

            pred_probs = pred_accum / (weight_accum + 1e-6)
            pred_mask = (pred_probs > 0.5).float()

        dice = dice_coefficient(pred_probs.unsqueeze(1), target_tensor.unsqueeze(1)).item()
        print(f"Dice score: {dice:.4f}")

        gt_sum = target_tensor[0].cpu().sum(dim=-1)
        pred_sum = pred_probs[0].cpu().sum(dim=-1)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(gt_sum, cmap='gray')
        axs[0].set_title("Ground Truth (sum Z)")
        axs[1].imshow(pred_sum, cmap='gray')
        axs[1].set_title("Prediction (sum Z)")
        plt.suptitle(f"Sample {i} - Dice: {dice:.4f}")
        plt.show()
