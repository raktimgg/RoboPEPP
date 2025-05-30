# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch # type: ignore
import torchvision.transforms as transforms # type: ignore

_GLOBAL_SEED = 0
logger = getLogger()

class ResizeLongerSide:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        # Determine the longer side and scale accordingly
        if w > h:
            new_w = self.crop_size
            new_h = int(self.crop_size * h / w)
        else:
            new_h = self.crop_size
            new_w = int(self.crop_size * w / h)
        return transforms.functional.resize(img, (new_h, new_w))

class PadToSquare:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        # Calculate padding needed to make the image square
        pad_w = (self.crop_size - w) // 2
        pad_h = (self.crop_size - h) // 2
        # Add padding to the image equally on all sides
        padding = (pad_w, pad_h, self.crop_size - w - pad_w, self.crop_size - h - pad_h)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='edge')


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    # transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    # transform_list += [transforms.Resize(crop_size)]
    # transform_list += [transforms.Pad(padding=(0, 0, 0, 0), fill=0, padding_mode='constant')]
    # transform_list += [transforms.CenterCrop(crop_size)]
    # transform_list += [ResizeLongerSide(crop_size)]
    # transform_list += [PadToSquare(crop_size)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        if isinstance(radius, torch.Tensor):
            radius = radius.item()  # Convert tensor to scalar if it's a single-value tensor

        return img.filter(ImageFilter.GaussianBlur(radius=radius))
