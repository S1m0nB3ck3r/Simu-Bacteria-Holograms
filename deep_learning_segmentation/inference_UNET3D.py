"""
Inference script for 3D U-Net hologram segmentation

Usage:
    python inference_UNET3D.py --model path/to/model.pth --holo_dir path/to/holograms --seg_dir path/to/segmentations
    python inference_UNET3D.py --model path/to/model.pth --holo_file path/to/holo.npy --seg_file path/to/seg.npy
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D, VolumeReconstructor, dice_coefficient


def load_model(model_path, device, dropout_prob=0.3):
    """Load trained model from checkpoint"""
    model = UNet3D(dropout_prob=dropout_prob).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def inference_on_volume(
    model, 
    volume_tensor, 
    device,
    patch_size_xy=(128, 128),
    patch_size_z=64,
    stride_xy=(64, 64),
    stride_z=32
):
    """
    Run inference on a full 3D volume using sliding window
    
    Args:
        model: Trained UNet3D model
        volume_tensor: Input volume of shape (1, X, Y, Z)
        device: torch device
        patch_size_xy: (width, height) of patches
        patch_size_z: Depth of patches
        stride_xy: (stride_x, stride_y) for sliding window
        stride_z: Stride along Z axis
        
    Returns:
        pred_probs: Probability map (1, X, Y, Z)
        pred_mask: Binary mask (1, X, Y, Z)
    """
    model.eval()
    volume_tensor = volume_tensor.to(device)
    
    X, Y, Z = volume_tensor.shape[1], volume_tensor.shape[2], volume_tensor.shape[3]
    patch_x, patch_y = patch_size_xy
    patch_z = patch_size_z
    stride_x, stride_y = stride_xy
    
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
    
    return pred_probs, pred_mask


def visualize_result(ground_truth, prediction, dice_score, save_path=None):
    """Visualize ground truth vs prediction"""
    gt_sum = ground_truth[0].cpu().sum(dim=-1).numpy()
    pred_sum = prediction[0].cpu().sum(dim=-1).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(gt_sum, cmap='gray')
    axs[0].set_title("Ground Truth (sum over Z)")
    axs[0].axis('off')
    
    axs[1].imshow(pred_sum, cmap='gray')
    axs[1].set_title("Prediction (sum over Z)")
    axs[1].axis('off')
    
    plt.suptitle(f"Dice Score: {dice_score:.4f}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def save_prediction(prediction_volume, output_path):
    """Save prediction as numpy file"""
    np.save(output_path, prediction_volume.cpu().numpy())
    print(f"Prediction saved to {output_path}")


def process_single_file(
    model,
    hologram_file,
    segmentation_file,
    parameters,
    device,
    output_dir=None,
    visualize=True,
    patch_size_xy=(128, 128),
    patch_size_z=64,
    stride_xy=(64, 64),
    stride_z=32
):
    """
    Process a single hologram file
    
    Args:
        model: Trained model
        hologram_file: Path to .npy hologram file
        segmentation_file: Path to .npy segmentation file
        parameters: Reconstruction parameters dict
        device: torch device
        output_dir: Directory to save results (optional)
        visualize: Whether to show visualization
        
    Returns:
        dice_score: Dice coefficient (if ground truth available)
    """
    print(f"\nProcessing: {os.path.basename(hologram_file)}")
    
    # Load data
    hologram_image = np.load(hologram_file)
    bool_volume_np = np.load(segmentation_file)
    
    # Reconstruct volume
    reconstructor = VolumeReconstructor(parameters)
    volume_tensor = reconstructor.volume_reconstruction(hologram_image, parameters)
    
    # Run inference
    pred_probs, pred_mask = inference_on_volume(
        model, volume_tensor, device,
        patch_size_xy=patch_size_xy,
        patch_size_z=patch_size_z,
        stride_xy=stride_xy,
        stride_z=stride_z
    )
    
    # Calculate Dice if ground truth available
    target_tensor = torch.from_numpy(bool_volume_np).float().unsqueeze(0).to(device)
    dice_score = dice_coefficient(pred_probs.unsqueeze(1), target_tensor.unsqueeze(1)).item()
    print(f"  Dice score: {dice_score:.4f}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(hologram_file))[0]
        
        # Save prediction
        pred_path = os.path.join(output_dir, f"{basename}_prediction.npy")
        save_prediction(pred_mask, pred_path)
        
        # Save visualization
        if visualize:
            vis_path = os.path.join(output_dir, f"{basename}_visualization.png")
            visualize_result(target_tensor, pred_probs, dice_score, save_path=vis_path)
    elif visualize:
        visualize_result(target_tensor, pred_probs, dice_score)
    
    return dice_score


def main():
    parser = argparse.ArgumentParser(description='3D U-Net Inference for Hologram Segmentation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--holo_dir', type=str, help='Path to directory with hologram files')
    parser.add_argument('--seg_dir', type=str, help='Path to directory with segmentation files')
    parser.add_argument('--holo_file', type=str, help='Path to single hologram file (.npy)')
    parser.add_argument('--seg_file', type=str, help='Path to single segmentation file (.npy)')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 64], 
                        help='Patch size (X Y Z)')
    parser.add_argument('--stride', type=int, nargs=3, default=[64, 64, 32],
                        help='Stride (X Y Z)')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = load_model(args.model, device)
    
    # Default reconstruction parameters
    parameters = {
        "holo_size_x": 512,
        "holo_size_y": 512,
        "holo_plane_number": 128,
        "medium_wavelength": 0.66e-6,
        "magnification_cam": 2.857,
        "pix_size_cam": 3.45e-6,
        "Z_step": 5e-6
    }
    
    # Prepare file pairs
    file_pairs = []
    if args.holo_file and args.seg_file:
        file_pairs.append((args.holo_file, args.seg_file))
    elif args.holo_dir and args.seg_dir:
        holo_files = sorted([f for f in os.listdir(args.holo_dir) if f.endswith('.npy')])
        for holo_file in holo_files:
            idx = holo_file.replace('holo_', '').replace('.npy', '')
            seg_file = f"segmentation_{idx}.npy"
            
            holo_path = os.path.join(args.holo_dir, holo_file)
            seg_path = os.path.join(args.seg_dir, seg_file)
            
            if os.path.exists(seg_path):
                file_pairs.append((holo_path, seg_path))
    else:
        print("Error: Must specify either --holo_file/--seg_file or --holo_dir/--seg_dir")
        return
    
    print(f"\nFound {len(file_pairs)} file pairs to process")
    
    # Process files
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)
    
    dice_scores = []
    patch_size_xy = (args.patch_size[0], args.patch_size[1])
    patch_size_z = args.patch_size[2]
    stride_xy = (args.stride[0], args.stride[1])
    stride_z = args.stride[2]
    
    for holo_path, seg_path in file_pairs:
        dice = process_single_file(
            model, 
            holo_path,
            seg_path,
            parameters,
            device, 
            output_dir=args.output_dir,
            visualize=not args.no_viz,
            patch_size_xy=patch_size_xy,
            patch_size_z=patch_size_z,
            stride_xy=stride_xy,
            stride_z=stride_z
        )
        dice_scores.append(dice)
    
    # Summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(file_pairs)} files")
    print(f"Average Dice score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    # If no arguments provided, use default configuration for testing
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Using default test configuration...")
        print("Usage: python inference_UNET3D.py --model MODEL_PATH --holo_dir HOLO_DIR --seg_dir SEG_DIR")
        print("\nRunning with default paths:")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Default paths
        model_path = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\models\unet3d_best.pth"
        test_holo_dir = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\data_split\test\simulated_hologram"
        test_seg_dir = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\data_split\test\binary_volume"
        output_dir = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\inference_results"
        
        if not os.path.exists(model_path):
            print(f"\nError: Model not found at {model_path}")
            print("Please train the model first or specify correct paths.")
            sys.exit(1)
        
        if not os.path.exists(test_holo_dir):
            print(f"\nError: Test directory not found at {test_holo_dir}")
            print("Please run split_data.py first.")
            sys.exit(1)
        
        # Load model
        print("\n" + "="*80)
        print("LOADING MODEL")
        print("="*80)
        model = load_model(model_path, device)
        
        # Reconstruction parameters
        parameters = {
            "holo_size_x": 512,
            "holo_size_y": 512,
            "holo_plane_number": 128,
            "medium_wavelength": 0.66e-6,
            "magnification_cam": 2.857,
            "pix_size_cam": 3.45e-6,
            "Z_step": 5e-6
        }
        
        # Get test file pairs
        holo_files = sorted([f for f in os.listdir(test_holo_dir) if f.endswith('.npy')])[:5]
        file_pairs = []
        for holo_file in holo_files:
            idx = holo_file.replace('holo_', '').replace('.npy', '')
            seg_file = f"segmentation_{idx}.npy"
            
            holo_path = os.path.join(test_holo_dir, holo_file)
            seg_path = os.path.join(test_seg_dir, seg_file)
            
            if os.path.exists(seg_path):
                file_pairs.append((holo_path, seg_path))
        
        print(f"\nProcessing {len(file_pairs)} test samples...")
        
        # Process files
        print("\n" + "="*80)
        print("INFERENCE")
        print("="*80)
        
        dice_scores = []
        for holo_path, seg_path in file_pairs:
            dice = process_single_file(
                model, holo_path, seg_path, parameters, device, 
                output_dir=output_dir, visualize=True
            )
            dice_scores.append(dice)
        
        # Summary
        print("\n" + "="*80)
        print("INFERENCE COMPLETE")
        print("="*80)
        print(f"\nProcessed {len(file_pairs)} files")
        print(f"Average Dice score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        
    else:
        main()
