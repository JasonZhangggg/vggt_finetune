import argparse
import numpy as np
import sys, os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Import your dataset class and VGGT model
from training.data.datasets.nuscenes import NuScenesDataset
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from types import SimpleNamespace


def compare_pose_encodings(pred_pose_enc, gt_pose_enc):
    """
    Compare predicted and ground truth pose encodings.
    
    Pose encoding format (9D):
    [0:3] - Translation (x, y, z)
    [3:7] - Rotation as quaternion (w, x, y, z)
    [7:9] - Field of view (horizontal, vertical)
    """
    print("\n" + "="*80)
    print("POSE ENCODING COMPARISON")
    print("="*80)
    
    pred = pred_pose_enc.cpu().numpy()
    gt = gt_pose_enc.cpu().numpy()
    
    # Compare each component
    trans_pred = pred[:, 0:3]
    trans_gt = gt[:, 0:3]
    trans_error = np.linalg.norm(trans_pred - trans_gt, axis=1)
    
    rot_pred = pred[:, 3:7]
    rot_gt = gt[:, 3:7]
    rot_error = np.linalg.norm(rot_pred - rot_gt, axis=1)
    
    fov_pred = pred[:, 7:9]
    fov_gt = gt[:, 7:9]
    fov_error = np.abs(fov_pred - fov_gt)
    
    print("\n📐 Translation Error (L2 norm per camera):")
    for i in range(len(trans_error)):
        print(f"  Camera {i}: {trans_error[i]:.6f}")
    print(f"  Mean: {trans_error.mean():.6f}, Max: {trans_error.max():.6f}")
    
    print("\n🔄 Rotation Error (L2 norm per camera):")
    for i in range(len(rot_error)):
        print(f"  Camera {i}: {rot_error[i]:.6f}")
    print(f"  Mean: {rot_error.mean():.6f}, Max: {rot_error.max():.6f}")
    
    print("\n📷 FoV Error (per camera):")
    for i in range(len(fov_error)):
        print(f"  Camera {i}: H={fov_error[i,0]:.6f}, V={fov_error[i,1]:.6f}")
    print(f"  Mean: {fov_error.mean():.6f}, Max: {fov_error.max():.6f}")
    
    # Check if coordinate systems match
    print("\n🔍 Coordinate System Check:")
    print("  First camera should be identity (reference frame)")
    print(f"  GT Translation [0]: {trans_gt[0]}")
    print(f"  GT Rotation [0]: {rot_gt[0]}")
    print(f"  Pred Translation [0]: {trans_pred[0]}")
    print(f"  Pred Rotation [0]: {rot_pred[0]}")
    
    # Overall assessment
    print("\n✅ Assessment:")
    if trans_error.mean() < 0.1 and rot_error.mean() < 0.1:
        print("  ✓ Predictions are in the SAME coordinate system as GT")
        print("  ✓ Model is ready for training")
    elif trans_error.mean() < 1.0 and rot_error.mean() < 1.0:
        print("  ⚠ Predictions have moderate errors")
        print("  ⚠ May need initialization or pre-training")
    else:
        print("  ✗ Large errors detected")
        print("  ✗ Coordinate systems may NOT match")
        print("  ✗ Check data preprocessing and model setup")
    
    return {
        'trans_error_mean': trans_error.mean(),
        'rot_error_mean': rot_error.mean(),
        'fov_error_mean': fov_error.mean(),
    }


def main(args):
    # Create a minimal config object to satisfy the dataset constructor
    common_conf = SimpleNamespace(
        debug=False,
        training=False,  # Set to False for testing
        get_nearby=False,
        load_depth=False,
        inside_random=False,
        allow_duplicate_img=False,
        img_size=518,
        patch_size=14,
        augs=SimpleNamespace(
            scales=[1.0, 1.0]
        ),
        rescale=True,  # Enable rescaling for proper preprocessing
        rescale_aug=False,
        landscape_check=True
    )
    
    # Initialize dataset
    print("Loading nuScenes dataset...")
    dataset = NuScenesDataset(
        common_conf=common_conf,
        split="mini_train",
        NUSCENES_DIR=args.data_root,
        version="v1.0-mini",
    )
    
    print(f"✓ Dataset loaded with {len(dataset)} samples.\n")
    
    # Load one sample
    print("Loading sample...")
    sample = dataset.get_data(seq_index=0)
    
    print("\n=== Sample Info ===")
    print(f"Images shape: {sample['images'].shape}")         # [6, H, W, 3]
    print(f"Extrinsics shape: {sample['extrinsics'].shape}") # [6, 4, 4]
    print(f"Intrinsics shape: {sample['intrinsics'].shape}") # [6, 3, 3]
    print(f"Point masks shape: {sample['point_masks'].shape}") # [6, H, W, 1]
    print(f"Image pixel range: {sample['images'].min():.2f} → {sample['images'].max():.2f}")
    
    # Convert to torch tensors and add batch dimension
    images = torch.from_numpy(sample['images']).permute(0, 3, 1, 2).float()  # [6, 3, H, W]
    images = images.unsqueeze(0)  # [1, 6, 3, H, W]
    
    extrinsics = torch.from_numpy(sample['extrinsics']).float()  # [6, 4, 4]
    intrinsics = torch.from_numpy(sample['intrinsics']).float()  # [6, 3, 3]
    
    # Normalize images to [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0
    
    print(f"\n✓ Converted to torch tensors")
    print(f"  Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Encode GT extrinsics and intrinsics to pose encoding
    print("\n=== Encoding Ground Truth ===")
    image_hw = torch.tensor([sample['images'].shape[1], sample['images'].shape[2]])
    
    gt_pose_encoding = extri_intri_to_pose_encoding(
        extrinsics.unsqueeze(0),  # [1, 6, 4, 4]
        intrinsics.unsqueeze(0),  # [1, 6, 3, 3]
        image_hw,
        pose_encoding_type="absT_quaR_FoV"
    )  # [1, 6, 9]
    
    print(f"✓ GT pose encoding shape: {gt_pose_encoding.shape}")
    print(f"\nGT pose encoding (first 2 cameras):")
    print(f"  Camera 0: {gt_pose_encoding[0, 0].numpy()}")
    print(f"  Camera 1: {gt_pose_encoding[0, 1].numpy()}")
    
    # Load VGGT model
    print("\n=== Loading VGGT Model ===")
    if args.pretrained_model:
        print(f"Loading from: {args.pretrained_model}")
        model = VGGT.from_pretrained(args.pretrained_model)
    else:
        print("Initializing model from scratch (no pretrained weights)")
        model = VGGT(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,
            enable_depth=False,
            enable_point=False,
            enable_track=False,
        )
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    images = images.to(device)
    gt_pose_encoding = gt_pose_encoding.to(device)
    
    print(f"✓ Model loaded on {device}")
    
    # Run inference
    print("\n=== Running VGGT Inference ===")
    with torch.no_grad():
        predictions = model(images)
    
    # Get predicted pose encoding
    pred_pose_enc = predictions['pose_enc']  # [1, 6, 9]
    
    print(f"✓ Predictions shape: {pred_pose_enc.shape}")
    print(f"\nPredicted pose encoding (first 2 cameras):")
    print(f"  Camera 0: {pred_pose_enc[0, 0].cpu().numpy()}")
    print(f"  Camera 1: {pred_pose_enc[0, 1].cpu().numpy()}")
    
    # Compare predictions with ground truth
    errors = compare_pose_encodings(pred_pose_enc[0], gt_pose_encoding[0])
    
    # Decode predictions back to extrinsics/intrinsics to verify
    print("\n=== Decoding Predictions ===")
    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
        pred_pose_enc,
        image_hw,
        pose_encoding_type="absT_quaR_FoV"
    )
    
    print(f"Predicted extrinsics shape: {pred_extrinsics.shape}")
    print(f"Predicted intrinsics shape: {pred_intrinsics.shape}")
    
    print("\n✓ Test completed successfully!")
    
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to nuScenes dataset root")
    args = parser.parse_args()
    main(args)
