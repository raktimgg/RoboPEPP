# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
import subprocess
import time
import cv2

import numpy as np
from PIL import Image

from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


class ResizeLongerSide:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        h, w = img.shape[:2]  # Get the height and width of the image
        if w > h:
            new_w = self.crop_size
            new_h = int(self.crop_size * h / w)
        else:
            new_h = self.crop_size
            new_w = int(self.crop_size * w / h)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized_img, (new_w, new_h)

class PadToSquare:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        h, w = img.shape[:2]  # Get the height and width of the image
        pad_h = (self.crop_size - h) // 2
        pad_w = (self.crop_size - w) // 2
        # Pad the image equally on all sides to make it square
        padding = ((pad_h, self.crop_size - h - pad_h), (pad_w, self.crop_size - w - pad_w), (0, 0))  # Last tuple for 3 channels (RGB)
        padded_img = np.pad(img, padding, mode='edge')
        return padded_img, (pad_w, pad_h)

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
    crop_size=224
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
        sampler=dist_sampler,
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
        copy_data=True,
        index_targets=False,
        crop_size=224
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

        # rel_folders = ['synthetic/panda_synth_train_dr', 'synthetic/panda_synth_test_photo', 
        #                'real/panda-3cam_azure', 'real/panda-3cam_kinect360', 'real/panda-3cam_realsense', 'real/anda-orb']
        rel_folders = ['synthetic/panda_synth_train_dr', 
                       'synthetic/kuka_synth_train_dr',
                       'synthetic/baxter_synth_train_dr']
        self.image_paths = []
        # in each of the rel_folders inside data_path, find all the images and add the locations to self.image_paths
        for rel_folder in rel_folders:
            folder = os.path.join(data_path, rel_folder)
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(root, file))

        self.resize_transform = ResizeLongerSide(crop_size)
        self.pad_transform = PadToSquare(crop_size)

        self.link_names = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 
                           'panda_link6', 'panda_link7', 'panda_hand',
                           'iiwa7_link_0', 'iiwa7_link_1', 'iiwa7_link_2', 'iiwa7_link_3',
                           'iiwa7_link_4', 'iiwa7_link_5', 'iiwa7_link_6', 'iiwa7_link_7',
                           'torso_t0', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0',
                           'left_w1', 'left_w2', 'left_hand', 'right_s0', 'right_s1', 'right_e0',
                           'right_e1', 'right_w0', 'right_w1', 'right_w2', 'right_hand']

        logger.info('Initialized DREAM dataset')
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        json_path = self.image_paths[index].replace('.rgb.jpg', '.json')
        with open(json_path, 'r') as jfile:
            annotations = json.load(jfile)
        bbox = annotations["objects"][0]["bounding_box"]
        # bbox_min = np.clip(np.array(bbox["min"]),0,1e6)
        # bbox_max = np.clip(np.array(bbox["max"]),0,1e6)
        bbox_min = np.clip(np.array(bbox["min"]),[0.0,0.0],[640.0,480.0])
        bbox_max = np.clip(np.array(bbox["max"]),[0.0,0.0],[640.0,480.0])
        img = np.array(img)[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]
        # print(bbox_min, bbox_max, img.shape, self.image_paths[index])
        img, (new_w, new_h) = self.resize_transform(img)
        img, (pad_w, pad_h) = self.pad_transform(img)
        img = Image.fromarray(img)

        keypoints = annotations['objects'][0]['keypoints']
        keypoints = sorted(keypoints, key=lambda kp: kp['name'])
        kp_data = []
        for kp in keypoints:
            if kp['name'] in self.link_names:
                kp_data.append(kp['projected_location'])
        
        keypoints = torch.as_tensor(kp_data)

        # print(keypoints.shape, bbox_min[0])

        keypoints[:, 0] -= bbox_min[0]  # Shift x-coordinates
        keypoints[:, 1] -= bbox_min[1]  # Shift y-coordinates

        scale_x = new_w / (bbox_max[0] - bbox_min[0])
        scale_y = new_h / (bbox_max[1] - bbox_min[1])
        keypoints[:, 0] *= scale_x  # Rescale x coordinates
        keypoints[:, 1] *= scale_y  # Rescale y coordinates

        keypoints[:, 0] += pad_w  # Adjust x coordinates for padding
        keypoints[:, 1] += pad_h  # Adjust y coordinates for padding

        target = keypoints
        if self.transform is not None:
            img = self.transform(img)
        return img, target