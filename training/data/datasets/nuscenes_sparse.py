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
from nuscenes.utils.data_classes import LidarPointCloud

import cv2
import numpy as np
from PIL import Image
from data.dataset_util import *
from data.base_dataset import BaseDataset
    
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion


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
        NUSCENES_DIR: str = "",
        version: str = "v1.0-trainval",
        len_train: int = 28130,
        len_test: int = 6019,
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

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
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
        all_cameras = [
            'CAM_FRONT',        # Reference camera (cam0)
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        
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
    
    def _project_lidar_to_camera(self, sample, cam_sd):
        """
        Project LiDAR points to camera image and create depth map.
        
        Args:
            sample: NuScenes sample dict
            cam_sd: Camera sample_data dict
            
        Returns:
            depth_map: (H, W) numpy array with depth values in meters
        """
        # Get LiDAR data (LIDAR_TOP)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_token)

        # Load LiDAR point cloud
        lidar_path = os.path.join(self.nusc.dataroot, lidar_sd['filename'])
        pc = LidarPointCloud.from_file(lidar_path)

        # Calibrations (sensor <-> ego)
        cam_calib = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)

        # Ego poses (ego -> global) at respective timestamps
        cam_ego_pose = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])

        # Build transforms as incremental rotations/translations applied to pc in-place
        # 1) LiDAR sensor -> Ego (lidar timestamp)
        pc.rotate(Quaternion(lidar_calib['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_calib['translation'], dtype=np.float64))

        # 2) Ego (lidar) -> Global
        pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
        pc.translate(np.array(lidar_ego_pose['translation'], dtype=np.float64))

        # 3) Global -> Ego (camera) [inverse of ego(cam)->global]
        R_ego_cam_global = Quaternion(cam_ego_pose['rotation']).rotation_matrix
        t_ego_cam_global = np.array(cam_ego_pose['translation'], dtype=np.float64)
        R_global_to_ego_cam = R_ego_cam_global.T
        t_global_to_ego_cam = -R_global_to_ego_cam @ t_ego_cam_global
        pc.rotate(R_global_to_ego_cam)
        pc.translate(t_global_to_ego_cam)

        # 4) Ego (camera) -> Camera [inverse of sensor->ego]
        R_cam_ego = Quaternion(cam_calib['rotation']).rotation_matrix
        t_cam_ego = np.array(cam_calib['translation'], dtype=np.float64)
        R_ego_to_cam = R_cam_ego.T
        t_ego_to_cam = -R_ego_to_cam @ t_cam_ego
        pc.rotate(R_ego_to_cam)
        pc.translate(t_ego_to_cam)

        # Points now in camera frame
        points = pc.points[:3, :]

        # Keep points in front of camera
        depths = points[2, :]
        valid = depths > 0
        points = points[:, valid]
        depths = depths[valid]

        # Project to image plane
        points_img = view_points(points, cam_intrinsic, normalize=True)

        # Use actual image dimensions from sample_data
        img_width = cam_sd.get('width', None)
        img_height = cam_sd.get('height', None)
        if img_width is None or img_height is None:
            # Fallback: read image to get size
            img_path = os.path.join(self.nusc.dataroot, cam_sd['filename'])
            with Image.open(img_path) as _im:
                img_width, img_height = _im.size

        # Filter points within image bounds
        u = points_img[0, :]
        v = points_img[1, :]
        in_bounds = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
        u = u[in_bounds]
        v = v[in_bounds]
        depths = depths[in_bounds]

        # Rasterize depth with z-buffer (nearest depth wins)
        depth_map = np.zeros((int(img_height), int(img_width)), dtype=np.float32)
        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)
        for uu, vv, d in zip(u_int, v_int, depths):
            # Bounds check (paranoia)
            if 0 <= uu < img_width and 0 <= vv < img_height:
                if depth_map[vv, uu] == 0 or d < depth_map[vv, uu]:
                    depth_map[vv, uu] = float(d)

        return depth_map

    def get_data(self, seq_index=None, img_per_seq=None, aspect_ratio=0.5625):
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
            images: list of (H, W, 3) uint8
            depths: list of (H, W) float32
            extrinsics: list of (3, 4) float32 (cam->first-cam)
            intrinsics: list of (3, 3) float32 (scaled for the processed image)
            cam_points: list of (H, W, 3) float32
            world_points: list of (H, W, 3) float32
            point_masks: list of (H, W) bool
            ids: np.ndarray of shape (S,), int64
        """
        if seq_index is None:
            seq_index = random.randint(0, len(self.samples) - 1)
        else:
            seq_index = seq_index % len(self.samples)
        
        sample_data = self.samples[seq_index]
        sample = sample_data['sample']
        cameras = sample_data['cameras']  # All 6 cameras

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        # First, get the reference frame (CAM_FRONT as reference)
        first_cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', first_cam_token)

        ego_to_cam0 = self.get_mat34_ego_to_cam(self.nusc, cam_data['calibrated_sensor_token'])
        cam0_to_ego = self.inv_extrinsic_34(ego_to_cam0)  # (cam0 -> ego)

        # Now process all 6 cameras
        for camera_idx, camera in enumerate(cameras):
            # Get camera data
            cam_token = sample['data'][camera]
            cam_sd = self.nusc.get('sample_data', cam_token)

            # Load image
            img_path = osp.join(self.NUSCENES_DIR, cam_sd['filename'])
            image = read_image_cv2(img_path)

            depth_map = self._project_lidar_to_camera(sample, cam_sd)

            assert image is not None and depth_map is not None, f"Failed to load image or depth for {img_path}"
            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            calib = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            intrinsic = np.array(calib['camera_intrinsic'], dtype=np.float32)

            ego_to_cam = self.get_mat34_ego_to_cam(self.nusc, cam_sd['calibrated_sensor_token'])
            cam0_to_cam = self.compose_34(ego_to_cam, cam0_to_ego)  # (cam0 -> cam)
            cam_to_cam0 = self.inv_extrinsic_34(cam0_to_cam)  # (cam -> cam0)

            processed_image, processed_depth, processed_extri, processed_intrinsic, \
                world_coords, cam_coords, point_mask, _ = self.process_one_image(
                    image=image,
                    depth_map=depth_map,
                    extri_opencv=cam_to_cam0,
                    intri_opencv=intrinsic,
                    original_size=original_size,
                    target_image_shape=target_image_shape,
                    filepath=img_path,
                )

            images.append(processed_image)
            depths.append(processed_depth)
            extrinsics.append(processed_extri)
            intrinsics.append(processed_intrinsic)
            cam_points.append(cam_coords)
            world_points.append(world_coords)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        # Prepare VKitti-style output (lists of numpy arrays). ComposedDataset will stack/convert.
        seq_name = f"nuscenes_{sample_data['scene']}_{sample['token']}"
        ids = np.arange(len(cameras), dtype=np.int64)

        return {
            'seq_name': seq_name,
            'ids': ids,
            'images': images,
            'depths': depths,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'cam_points': cam_points,
            'world_points': world_points,
            'point_masks': point_masks,
            'original_sizes': original_sizes,
        }
