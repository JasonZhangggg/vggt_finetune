# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Fine-tune VGGT on nuScenes dataset using camera head loss only.
Freezes all parameters except the last 2 blocks of alternating attention.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from training.loss import MultitaskLoss
from training.data.datasets.nuscenes import NuScenesDataset
from training.train_utils.freeze import freeze_modules
from training.train_utils.optimizer import build_optimizer
from training.train_utils.logging import setup_logging
from training.train_utils.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def freeze_except_last_n_aa_blocks(model: VGGT, n_blocks: int = 2):
    """
    Freeze all parameters in VGGT except the last n alternating attention blocks.
    
    Args:
        model: VGGT model instance
        n_blocks: Number of alternating attention block pairs to keep trainable
        
    Returns:
        model with frozen parameters
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the aggregator's blocks
    aggregator = model.aggregator
    depth = aggregator.depth
    aa_block_size = aggregator.aa_block_size
    
    # Calculate how many blocks to unfreeze
    # We want to unfreeze the last n_blocks * aa_block_size actual transformer blocks
    blocks_to_unfreeze = n_blocks * aa_block_size
    
    # Unfreeze last blocks from frame_blocks
    frame_start_idx = max(0, len(aggregator.frame_blocks) - blocks_to_unfreeze)
    for idx in range(frame_start_idx, len(aggregator.frame_blocks)):
        for param in aggregator.frame_blocks[idx].parameters():
            param.requires_grad = True
    
    # Unfreeze last blocks from global_blocks
    global_start_idx = max(0, len(aggregator.global_blocks) - blocks_to_unfreeze)
    for idx in range(global_start_idx, len(aggregator.global_blocks)):
        for param in aggregator.global_blocks[idx].parameters():
            param.requires_grad = True
    
    # Log which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    logger.info(f"Unfrozen frame blocks: {frame_start_idx} to {len(aggregator.frame_blocks) - 1}")
    logger.info(f"Unfrozen global blocks: {global_start_idx} to {len(aggregator.global_blocks) - 1}")
    
    return model


def build_dataloaders(cfg: DictConfig):
    """
    Build training and validation dataloaders.
    
    Args:
        cfg: Configuration object
        
    Returns:
        train_loader, val_loader
    """
    # Create common config for datasets
    common_conf = OmegaConf.create({
        'img_size': cfg.data.img_size,
        'patch_size': cfg.model.patch_size,
        'augs': {'scales': cfg.data.aug_scales},
        'rescale': cfg.data.rescale,
        'rescale_aug': cfg.data.rescale_aug,
        'landscape_check': cfg.data.landscape_check,
        'debug': cfg.debug,
        'training': True,
        'get_nearby': False,
        'load_depth': False,
        'inside_random': cfg.data.inside_random,
        'allow_duplicate_img': False,
    })
    
    # Training dataset
    # NOTE: Dataset automatically loads all 6 simultaneous cameras per sample
    # Each batch gets batch_size samples × 6 cameras = batch_size * 6 total images
    train_dataset = NuScenesDataset(
        common_conf=common_conf,
        split='train',
        NUSCENES_DIR=cfg.data.nuscenes_dir,
        camera_channels=cfg.data.camera_channels,
        version=cfg.data.version,
        len_train=cfg.data.len_train,
    )
    
    # Validation dataset
    val_common_conf = OmegaConf.create(common_conf)
    val_common_conf.training = False
    val_common_conf.inside_random = False
    
    val_dataset = NuScenesDataset(
        common_conf=val_common_conf,
        split='val',
        NUSCENES_DIR=cfg.data.nuscenes_dir,
        camera_channels=cfg.data.camera_channels,
        version=cfg.data.version,
        len_train=cfg.data.len_test,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, cfg):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_camera = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch_data in enumerate(pbar):
        # Move data to device
        # batch_data is a dict with numpy arrays, need to convert and move
        batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, np.ndarray):
                batch[key] = torch.from_numpy(value).to(device)
            else:
                batch[key] = value
        
        # Transpose images from [B, S, H, W, 3] to [B, S, 3, H, W]
        if batch['images'].dim() == 5 and batch['images'].shape[-1] == 3:
            batch['images'] = batch['images'].permute(0, 1, 4, 2, 3)
        
        # Transpose point_masks from [B, S, H, W, 1] to [B, S, 1, H, W]
        if batch['point_masks'].dim() == 5 and batch['point_masks'].shape[-1] == 1:
            batch['point_masks'] = batch['point_masks'].permute(0, 1, 4, 2, 3)
        
        optimizer.zero_grad()
        
        # Forward pass
        try:
            with torch.cuda.amp.autocast(enabled=cfg.training.use_amp):
                predictions = model(batch['images'])
                
                # Compute loss
                loss_dict = loss_fn(predictions, batch)
                loss = loss_dict['objective']
            
            # Backward pass
            if cfg.training.use_amp:
                scaler = train_epoch.scaler
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if cfg.training.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=cfg.training.grad_clip
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if cfg.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=cfg.training.grad_clip
                    )
                
                optimizer.step()
            
            total_loss += loss.item()
            total_loss_camera += loss_dict.get('loss_camera', 0).item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'loss_cam': loss_dict.get('loss_camera', 0).item()
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_camera = total_loss_camera / len(dataloader)
    
    return {
        'loss': avg_loss,
        'loss_camera': avg_loss_camera,
    }


# Initialize scaler for AMP
train_epoch.scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, cfg):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_loss_camera = 0
    
    for batch_data in tqdm(dataloader, desc="Validation"):
        # Move data to device
        batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, np.ndarray):
                batch[key] = torch.from_numpy(value).to(device)
            else:
                batch[key] = value
        
        # Transpose images and masks
        if batch['images'].dim() == 5 and batch['images'].shape[-1] == 3:
            batch['images'] = batch['images'].permute(0, 1, 4, 2, 3)
        if batch['point_masks'].dim() == 5 and batch['point_masks'].shape[-1] == 1:
            batch['point_masks'] = batch['point_masks'].permute(0, 1, 4, 2, 3)
        
        try:
            with torch.cuda.amp.autocast(enabled=cfg.training.use_amp):
                predictions = model(batch['images'])
                loss_dict = loss_fn(predictions, batch)
                loss = loss_dict['objective']
            
            total_loss += loss.item()
            total_loss_camera += loss_dict.get('loss_camera', 0).item()
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            continue
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_camera = total_loss_camera / len(dataloader) if len(dataloader) > 0 else 0
    
    return {
        'loss': avg_loss,
        'loss_camera': avg_loss_camera,
    }


@hydra.main(version_base=None, config_path="config", config_name="finetune_camera")
def main(cfg: DictConfig):
    """Main training function."""
    # Setup
    logger.info("="*80)
    logger.info("Fine-tuning VGGT Camera Head on nuScenes")
    logger.info("="*80)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained VGGT model
    logger.info(f"Loading pretrained VGGT model from {cfg.model.pretrained}...")
    model = VGGT.from_pretrained(cfg.model.pretrained)
    model = model.to(device)
    
    # Freeze all except last n alternating attention blocks
    logger.info(f"Freezing all parameters except last {cfg.model.n_blocks_unfreeze} AA blocks...")
    model = freeze_except_last_n_aa_blocks(model, n_blocks=cfg.model.n_blocks_unfreeze)
    
    # Setup loss function (camera loss only)
    loss_fn = MultitaskLoss(
        camera={
            'weight': cfg.loss.camera.weight,
            'loss_type': cfg.loss.camera.loss_type,
            'gamma': cfg.loss.camera.gamma,
            'pose_encoding_type': cfg.loss.camera.pose_encoding_type,
            'weight_trans': cfg.loss.camera.weight_trans,
            'weight_rot': cfg.loss.camera.weight_rot,
            'weight_focal': cfg.loss.camera.weight_focal,
        }
    )
    
    # Setup dataloaders
    logger.info("Setting up dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
        eta_min=cfg.training.lr * cfg.training.lr_min_ratio,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    if cfg.resume_from is not None and Path(cfg.resume_from).exists():
        logger.info(f"Resuming from checkpoint: {cfg.resume_from}")
        checkpoint = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{cfg.training.num_epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, cfg)
        logger.info(f"Train Loss: {train_metrics['loss']:.6f}, "
                   f"Camera Loss: {train_metrics['loss_camera']:.6f}")
        
        # Validate
        if epoch % cfg.training.val_every == 0:
            val_metrics = validate(model, val_loader, loss_fn, device, cfg)
            logger.info(f"Val Loss: {val_metrics['loss']:.6f}, "
                       f"Camera Loss: {val_metrics['loss_camera']:.6f}")
        else:
            val_metrics = {'loss': float('inf'), 'loss_camera': float('inf')}
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6e}")
        
        # Save checkpoint
        if epoch % cfg.training.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'cfg': OmegaConf.to_container(cfg),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'cfg': OmegaConf.to_container(cfg),
            }, best_path)
            logger.info(f"✓ New best model saved with val_loss: {val_metrics['loss']:.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info("="*80)


if __name__ == "__main__":
    import numpy as np
    main()
