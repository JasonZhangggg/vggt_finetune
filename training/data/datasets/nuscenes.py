# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
try:
    from data.dataset_util import *
    from data.base_dataset import BaseDataset
except ModuleNotFoundError:
    # If running from notebook or different location
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from training.data.dataset_util import *
    from training.data.base_dataset import BaseDataset
    
# You'll need: pip install nuscenes-devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    logging.warning("nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")


class NuScenesDataset(BaseDataset):
    """
    nuScenes dataset for camera pose estimation.
    
    This dataset loads sequences of images from nuScenes with ground truth
    camera extrinsics and intrinsics for training VGGT camera head.
    """
    
    def __init__(
        self,
        common_conf,
        split: str = "train",
        NUSCENES_DIR: str = None,
        version: str = "v1.0-trainval",
        len_train: int = 50000,
        len_test: int = 5000,
    ):
        """
        Initialize the NuScenesDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'/'val'.
            NUSCENES_DIR (str): Directory path to nuScenes data.
            version (str): nuScenes version to load.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            
        Raises:
            ValueError: If NUSCENES_DIR is not specified or nuscenes-devkit not installed.
        """
        super().__init__(common_conf=common_conf)

        if not NUSCENES_AVAILABLE:
            raise ValueError(
                "nuscenes-devkit is not installed. "
                "Install with: pip install nuscenes-devkit"
            )

        if NUSCENES_DIR is None:
            raise ValueError("NUSCENES_DIR must be specified.")

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.NUSCENES_DIR = NUSCENES_DIR

        # Set dataset length based on split
        if split == "train":
            self.len_train = len_train
        elif split in ["test", "val"]:
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"NUSCENES_DIR is {NUSCENES_DIR}")
        logging.info(f"Loading nuScenes {version} {split} split...")

        # Initialize nuScenes
        self.nusc = NuScenes(version=version, dataroot=NUSCENES_DIR, verbose=True)
        
        # Build sample list (each sample has 6 simultaneous camera views)
        self.samples = self._build_samples(split)
        
        logging.info(f"Loaded {len(self.samples)} samples from nuScenes (each with 6 cameras)")

    def _build_samples(self, split: str) -> List[Dict]:
        """
        Build list of samples from nuScenes scenes.
        Each sample contains 6 simultaneous camera views.
        
        Args:
            split: Dataset split name
            
        Returns:
            List of sample dictionaries, each containing a single timestep with 6 cameras
        """
        samples = []
        
        # Get scenes for the split
        split_scenes = self._get_split_scenes(split)
        
        # All available cameras in nuScenes (simultaneous views)
        all_cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                      'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        for scene in self.nusc.scene:
            if scene['name'] not in split_scenes:
                continue
                
            # Get all samples in this scene
            sample_token = scene['first_sample_token']
            
            while sample_token != '':
                sample = self.nusc.get('sample', sample_token)
                
                # Each sample has 6 simultaneous cameras - this is one data point
                samples.append({
                    'scene': scene['name'],
                    'sample': sample,
                    'cameras': all_cameras,  # All 6 cameras at this timestep
                })
                
                sample_token = sample['next']
        
        return samples
    
    def _get_split_scenes(self, split: str) -> List[str]:
        """
        Get scene names for the split.
        
        Args:
            split: Dataset split name
            
        Returns:
            List of scene names
        """
        # nuScenes official split
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        
        if split == "train":
            return splits['train']
        elif split in ["val", "test"]:
            return splits['val']
        else:
            # Fallback
            all_scenes = [s['name'] for s in self.nusc.scene]
            n_train = int(0.8 * len(all_scenes))
            if split == 'train':
                return all_scenes[:n_train]
            else:
                return all_scenes[n_train:]
    
    def __len__(self):
        return self.len_train
    
    def mat34_to_Rt(self, E):
        return E[:, :3], E[:, 3]

    def get_mat34_ego_to_cam(self, nusc, token: str) -> np.ndarray:
        calib = nusc.get('calibrated_sensor', token)
        R_se = Quaternion(calib['rotation']).rotation_matrix   # cam->ego
        t_se = np.array(calib['translation'])
        
        # Convert to (ego -> cam) and create (3,4) extrinsic matrix
        E = np.zeros((3,4), dtype=np.float64)
        E[:, :3] = R_se.T
        E[:, 3]  = -R_se.T @ t_se # ego->cam
        return E

    def inv_extrinsic_34(self, E):
        R, t = self.mat34_to_Rt(E)
        Ei = np.zeros((3,4), dtype=np.float64)
        Ei[:, :3] = R.T
        Ei[:, 3]  = -R.T @ t
        return Ei
    
    def compose_34(self, A, B):
        RA, tA = self.mat34_to_Rt(A)
        RB, tB = self.mat34_to_Rt(B)
        RC = RA @ RB
        tC = RA @ tB + tA
        C = np.zeros((3,4), dtype=np.float64)
        C[:, :3] = RC
        C[:, 3]  = tC
        return C

    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=1.0):
        """
        Get data for a given sample with all 6 simultaneous cameras.
        
        IMPORTANT: 
        1. Converts nuScenes ego-relative coordinates to VGGT's first-camera-relative coordinates.
        2. Properly scales intrinsics for VGGT's cropped images (not just resized).
        
        Args:
            seq_index: Index of the sample to load
            img_per_seq: Not used (kept for compatibility)
            aspect_ratio: Target aspect ratio
            
        Returns:
            Dictionary containing:
                - images: [6, H, W, 3] - 6 cameras at single timestep
                - extrinsics: [6, 4, 4] - camera-to-first-camera transforms
                - intrinsics: [6, 3, 3] - camera intrinsics (properly scaled for cropped images)
                - point_masks: [6, H, W, 1] - all ones (valid mask)
                - depths: Optional [6, H, W, 1] if load_depth is True
        """
        if seq_index is None:
            seq_index = random.randint(0, len(self.samples) - 1)
        else:
            seq_index = seq_index % len(self.samples)
        
        sample_data = self.samples[seq_index]
        sample = sample_data['sample']
        cameras = sample_data['cameras']  # All 6 cameras
        
        # Get target shape (what VGGT expects after cropping)
        target_height, target_width = self.get_target_shape(aspect_ratio)
        
        # Load images and camera parameters for ALL 6 cameras
        images_list = []
        extrinsics_list = []
        intrinsics_list = []
        depths_list = [] if self.load_depth else None
        
        # First, get the reference frame (CAM_FRONT as reference)
        first_cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', first_cam_token)

        ego_to_cam0 = self.get_mat34_ego_to_cam(self.nusc, cam_data['calibrated_sensor_token'])
        cam0_to_ego = self.inv_extrinsic_34(ego_to_cam0)  # (cam0 -> ego)

        # Now process all 6 cameras
        for camera_idx, camera in enumerate(cameras):
            # Get camera data
            cam_token = sample['data'][camera]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            # Load image
            img_path = osp.join(self.NUSCENES_DIR, cam_data['filename'])
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Failed to load image: {img_path}")
                # Use blank image as fallback
                image = np.zeros((900, 1600, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get calibration
            calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            # Original intrinsics
            intrinsic = np.array(calib['camera_intrinsic'], dtype=np.float32)
            
            # Original image size
            original_height, original_width = 900, 1600  # nuScenes default
            original_size = np.array([original_height, original_width])
            
            # Extrinsics: Convert from ego-relative to first-camera-relative
            ego_to_cam = self.get_mat34_ego_to_cam(self.nusc, cam_data['calibrated_sensor_token'])
            
            cam0_to_cam = self.compose_34(ego_to_cam, cam0_to_ego)  # (cam0 -> cam)

            # Depth (placeholder, nuScenes doesn't have dense depth)
            depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
            # Use process_one_image to handle proper resizing, cropping, and intrinsics adjustment
            # This ensures intrinsics match VGGT's expected format
            target_shape = np.array([target_height, target_width])
            
            processed_image, processed_depth, cam0_to_cam, processed_intrinsic, \
                world_coords, cam_coords, point_mask, _ = self.process_one_image(
                    image=image,
                    depth_map=depth,
                    extri_opencv=cam0_to_cam,
                    intri_opencv=intrinsic,
                    original_size=original_size,
                    target_image_shape=target_shape,
                    track=None,
                    filepath=None,
                )
            
            images_list.append(processed_image)
            extrinsics_list.append(cam0_to_cam)
            intrinsics_list.append(processed_intrinsic)
            
            if self.load_depth:
                depths_list.append(processed_depth)
        
        # Stack arrays
        images = np.stack(images_list, axis=0)  # [6, H, W, 3]
        extrinsics = np.stack(extrinsics_list, axis=0)  # [6, 4, 4]
        intrinsics = np.stack(intrinsics_list, axis=0)  # [6, 3, 3]
        
        # Create point masks (all valid for nuScenes)
        point_masks = np.ones((6, target_height, target_width, 1), dtype=np.float32)
        
        # Convert to dictionary format expected by trainer
        data_dict = {
            'images': images,  # [6, H, W, 3]
            'extrinsics': extrinsics,  # [6, 4, 4] - relative to CAM_FRONT
            'intrinsics': intrinsics,  # [6, 3, 3] - properly scaled for cropped images
            'point_masks': point_masks,  # [6, H, W, 1]
        }
        
        if self.load_depth:
            depths = np.stack(depths_list, axis=0)  # [6, H, W, 1]
            data_dict['depths'] = depths
        
        return data_dict
