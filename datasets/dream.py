# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import json
import os
import random
from typing import OrderedDict

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from logging import getLogger
from .augmentations import PadToSquare, PillowSharpness, PillowContrast, PillowBrightness, PillowColor, apply_color_jitter, apply_occlusion, apply_rgb_aug, occlusion_aug, ResizeLongerSide
from .image_proc import create_belief_map

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


def make_dream(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    crop_size = 224
):
    dataset = DREAM(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False,
        crop_size=crop_size)
    logger.info('Dream dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('Dream unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class DREAM(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=False,
        index_targets=False,
        crop_size = 224,
        robot_name='panda'
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        data_path = None
        self.transform = transform

        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder)
        logger.info(f'data-path {data_path}')
        self.train = train
        self.color_jitter = True
        self.rgb_augmentation = True
        self.occlusion_augmentation= True
        self.crop_size = crop_size
        self.resize_transform = ResizeLongerSide(crop_size)
        self.pad_transform = PadToSquare(crop_size)
        self.occlu_p = 0.5

        if train:
            if robot_name=='panda':
                rel_folders = ['synthetic/panda_synth_train_dr']
            elif robot_name=='kuka':
                rel_folders = ['synthetic/kuka_synth_train_dr']
        else:
            if robot_name=='panda':
                rel_folders = ['synthetic/panda_synth_test_dr']
            elif robot_name=='kuka':
                rel_folders = ['synthetic/kuka_synth_test_dr']
        self.image_paths = []

        for rel_folder in rel_folders:
            folder = os.path.join(data_path, rel_folder)
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(root, file))


        if 'panda' in self.image_paths[0]:
            self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1']
            self.link_names = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 
                            'panda_link6', 'panda_link7', 'panda_hand']
        if 'kuka' in self.image_paths[0]:
            self.joint_names = ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 
                'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']
            self.link_names = ['iiwa7_link_0', 'iiwa7_link_1', 'iiwa7_link_2', 'iiwa7_link_3',
                                'iiwa7_link_4', 'iiwa7_link_5', 'iiwa7_link_6', 'iiwa7_link_7']
            # there is a discrepency between the link names here and the link names in urdf
            # iiwa7_* in json is iiwa_* in urdf
        if 'baxter' in self.image_paths[0]:
            self.joint_names = ['head_pan', 'right_s0', 'left_s0', 'right_s1', 'left_s1', 
                            'right_e0', 'left_e0', 'right_e1', 'left_e1', 'right_w0', 
                            'left_w0', 'right_w1', 'left_w1', 'right_w2', 'left_w2']
            self.link_names = ['torso_t0', 'right_s0','left_s0', 'right_s1', 'left_s1',
                        'right_e0','left_e0', 'right_e1','left_e1','right_w0', 'left_w0',
                        'right_w1','left_w1','right_w2', 'left_w2','right_hand','left_hand']
            
        logger.info('Initialized DREAM dataset')

        self.epoch = 0
    
    def __len__(self):
        return len(self.image_paths)
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        json_path = self.image_paths[index].replace('.rgb.jpg', '.json')
        with open(json_path, 'r') as jfile:
            annotations = json.load(jfile)
        joints = annotations['sim_state']['joints']
        joints = OrderedDict({d['name'].split('/')[-1]: float(d['position']) for d in joints})
        if 'kuka' in self.image_paths[index]:
            joints = {k.replace('iiwa7_', 'iiwa_'): v for k,v in joints.items()}
        jointpose = torch.as_tensor([joints[k] for k in self.joint_names])

        keypoints = annotations['objects'][0]['keypoints']
        keypoints = sorted(keypoints, key=lambda kp: kp['name'])
        name_data = []
        kp_data = []
        kp3d_data = []

        # Create a dictionary to quickly access keypoints by name
        keypoint_dict = {kp['name']: kp for kp in keypoints}

        # Iterate through self.link_names in the specified order
        for link_name in self.link_names:
            if link_name in keypoint_dict:
                kp = keypoint_dict[link_name]
                name_data.append(kp['name'])
                kp_data.append(kp['projected_location'])
                kp3d_data.append(kp['location'])

        keypoints = torch.as_tensor(kp_data)
        keypoints_3d = torch.as_tensor(kp3d_data)
        keypoints_3d = keypoints_3d/100
        
        bbox = annotations["objects"][0]["bounding_box"]
        bbox_min = np.array(bbox["min"])
        bbox_max = np.array(bbox["max"])


        if self.epoch < 30:
            bbox_min = bbox_min
            bbox_max = bbox_max
        elif self.epoch < 50:
            bbox_min = bbox_min - (np.random.rand(2))*30
            bbox_max = bbox_max + (np.random.rand(2))*30
        elif self.epoch < 70:
            bbox_min = bbox_min - (np.random.rand(2))*50
            bbox_max = bbox_max + (np.random.rand(2))*50
        elif self.epoch < 90:
            bbox_min = bbox_min - (np.random.rand(2))*80
            bbox_max = bbox_max + (np.random.rand(2))*80
        elif self.epoch < 110:
            bbox_min = bbox_min - (np.random.rand(2))*100
            bbox_max = bbox_max + (np.random.rand(2))*100
        elif self.epoch < 150:
            bbox_min = bbox_min - (np.random.rand(2))*120
            bbox_max = bbox_max + (np.random.rand(2))*120
        else:
            bbox_min = bbox_min - (np.random.rand(2))*120
            bbox_max = bbox_max + (np.random.rand(2))*120


        bbox_min = np.clip(bbox_min,[0.0,0.0],[640.0,480.0])
        bbox_max = np.clip(bbox_max,[0.0,0.0],[640.0,480.0])

        metadata = {}
        metadata['img_path'] = self.image_paths[index]
        metadata['orig_img'] = copy.deepcopy(np.array(img))
        metadata['orig_keypoints'] = copy.deepcopy(keypoints.numpy())
        metadata['orig_keypoints_3d'] = copy.deepcopy(keypoints_3d.numpy())

        img = np.array(img)[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]

        # Adjust keypoints to account for cropping (shift them relative to bbox_min)
        keypoints[:, 0] -= bbox_min[0]  # Shift x-coordinates
        keypoints[:, 1] -= bbox_min[1]  # Shift y-coordinates
        metadata['bbox_min'] = bbox_min
        metadata['bbox_max'] = bbox_max
        
        img, (new_w, new_h) = self.resize_transform(img)

        # Rescale the keypoints according to the new width and height
        scale_x = new_w / (bbox_max[0] - bbox_min[0])
        scale_y = new_h / (bbox_max[1] - bbox_min[1])
        keypoints[:, 0] *= scale_x  # Rescale x coordinates
        keypoints[:, 1] *= scale_y  # Rescale y coordinates
        metadata['scale'] = (scale_x, scale_y)

        img, (pad_w, pad_h) = self.pad_transform(img)
        # Adjust keypoints based on the padding applied
        keypoints[:, 0] += pad_w  # Adjust x coordinates for padding
        keypoints[:, 1] += pad_h  # Adjust y coordinates for padding
        metadata['pad'] = (pad_w, pad_h)

        valid_indices_mask =(keypoints[:, 0] > 0) & (keypoints[:, 0] < self.crop_size) & \
                    (keypoints[:, 1] > 0) & (keypoints[:, 1] < self.crop_size)
        metadata['valid_indices_mask'] = valid_indices_mask
        img = Image.fromarray(img)

        belief_maps = create_belief_map(
            image_resolution = (self.crop_size, self.crop_size), pointsBelief = keypoints.numpy(), sigma = 2
        )
        belief_maps_as_tensor = torch.tensor(belief_maps).float()

        if self.train:
            rgb = np.array(img)
            if self.color_jitter and random.random()<0.4:#color jitter #0.4
                rgb = apply_color_jitter(rgb)
                
            if self.occlusion_augmentation and random.random() < self.occlu_p: #0.5
                rgb = apply_occlusion(rgb)
                    
            if self.rgb_augmentation :
                rgb = apply_rgb_aug(rgb)
            img = rgb

        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # Intrinsic parameters of the camera
        # camera_settings.jsonn is in the same folder as the jpg and other json files
        came_json_path = os.path.join(os.path.dirname(json_path), '_camera_settings.json')
        with open(came_json_path) as f:
            camera_settings = json.load(f)['camera_settings'][0]
        fx = camera_settings['intrinsic_settings']['fx'] 
        fy = camera_settings['intrinsic_settings']['fy']
        cx = camera_settings['intrinsic_settings']['cx']
        cy = camera_settings['intrinsic_settings']['cy']

        intrinsics = np.array([[   fx,    0.     ,    cx   ],
                                [   0.     ,    fy,    cy   ],
                                [   0.     ,    0.     ,    1.        ]])
        metadata['K'] = intrinsics

        R_NORMAL_UE = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ])
        # get camera pose in translation and quertnion and convert to transformation matrix 
        camera_pos = np.array(annotations['camera_data']['location_worldframe'])
        camera_quat = np.array(annotations['camera_data']['quaternion_xyzw_worldframe'])
        # Convert quaternion to rotation matrix
        rotation = R.from_quat(camera_quat)  # from_quat expects [x, y, z, w] format
        rotation_matrix = rotation.as_matrix()@R_NORMAL_UE  # Convert to 3x3 rotation matrix
        # Create the 4x4 transformation matrix
        camera_pose = np.eye(4)  # Start with identity matrix
        camera_pose[:3, :3] = rotation_matrix  # Set the top-left 3x3 part as the rotation matrix
        camera_pose[:3, 3] = camera_pos  # Set the top-right 3x1 part as the translation vector
        metadata['wTc'] = np.array(camera_pose)/100

        robot_pos = annotations['objects'][0]['location']
        robot_quat = annotations['objects'][0]['quaternion_xyzw']
        # Convert quaternion to rotation matrix
        rotation = R.from_quat(robot_quat)
        rotation_matrix = rotation.as_matrix()@R_NORMAL_UE
        # Create the 4x4 transformation matrix
        robot_pose = np.eye(4)
        robot_pose[:3, :3] = rotation_matrix
        robot_pose[:3, 3] = np.array(robot_pos)/100
        metadata['cTr'] = robot_pose

        return img, jointpose, belief_maps_as_tensor, metadata