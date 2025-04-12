import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
from copy import deepcopy
import numpy as np
import PIL
import torch
import torch.nn.functional as F # type: ignore
from PIL import ImageEnhance, ImageFilter, Image
import cv2


def to_pil(im):
    if isinstance(im, PIL.Image.Image):
        return im
    elif isinstance(im, torch.Tensor):
        return PIL.Image.fromarray(np.asarray(im))
    elif isinstance(im, np.ndarray):
        return PIL.Image.fromarray(im)
    else:
        raise ValueError('Type not supported', type(im))


def to_torch_uint8(im):
    if isinstance(im, PIL.Image.Image):
        im = torch.as_tensor(np.asarray(im).astype(np.uint8))
    elif isinstance(im, torch.Tensor):
        assert im.dtype == torch.uint8
    elif isinstance(im, np.ndarray):
        assert im.dtype == np.uint8
        im = torch.as_tensor(im)
    else:
        raise ValueError('Type not supported', type(im))
    if im.dim() == 3:
        assert im.shape[-1] in {1, 3},f"{im.shape}"
    return im

def occlusion_aug(bbox, img_shape, min_area=0.0, max_area=0.3, max_try_times=5):
    # xmin, ymin, _, _ = bbox
    # xmax = bbox[2]
    # ymax = bbox[3]
    imght, imgwidth = img_shape
    xmin, ymin, xmax, ymax = 0,0,imgwidth,imght
    counter = 0
    while True:
        # force to break if no suitable occlusion
        if counter > max_try_times: # 5
            print('No suitable occlusion')
            return 0, 0, 0, 0
        counter += 1

        area_min = min_area # 0.0
        area_max = max_area # 0.3
        synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)
        
        ratio_min = 0.5
        ratio_max = 1 / 0.5
        synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

        if(synth_ratio*synth_area<=0):
            print(synth_area,xmax,xmin,ymax,ymin)
            print(synth_ratio,ratio_max,ratio_min)           
        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            break
    return synth_ymin, synth_h, synth_xmin, synth_w

class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        k = random.randint(*self.factor_interval)
        im = im.filter(ImageFilter.GaussianBlur(k))
        return im, mask, obs


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        if random.random() <= self.p:
            im = self._pillow_fn(im).enhance(factor=random.uniform(*self.factor_interval))
        #im.save('./BRIGHT.png')
        return im, mask, obs


class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0., 50.)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval)


class GrayScale(PillowRGBAugmentation):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        if random.random() <= self.p:
            im = to_torch_uint8(im).float()
            gray = 0.2989 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2]
            gray = gray.to(torch.uint8)
            im = gray.unsqueeze(-1).repeat(1, 1, 3)
        return im, mask, obs


class BackgroundAugmentation:
    def __init__(self, image_dataset, p):
        self.image_dataset = image_dataset
        self.p = p

    def get_bg_image(self, idx):
        return self.image_dataset[idx]

    def __call__(self, im, mask, obs):
        if random.random() <= self.p:
            im = to_torch_uint8(im)
            mask = to_torch_uint8(mask)
            h, w, c = im.shape
            im_bg = self.get_bg_image(random.randint(0, len(self.image_dataset) - 1))
            im_bg = to_pil(im_bg)
            im_bg = torch.as_tensor(np.asarray(im_bg.resize((w, h))))
            mask_bg = mask == 0
            im[mask_bg] = im_bg[mask_bg]
        return im, mask, obs


def flip_img(x):
    assert (x.ndim == 3)
    dim = 1
    return np.flip(x,axis=dim).copy()

def flip_joints_2d(joints_2d, width, flip_pairs):
    joints = joints_2d.copy()
    joints[:, 0] = width - joints[:, 0] - 1 # flip horizontally
    
    if flip_pairs is not None: # change left-right parts
        for lr in flip_pairs:
            joints[lr[0]], joints[lr[1]] = joints[lr[1]].copy(), joints[lr[0]].copy()
    return joints

def flip_xyz_joints_3d(joints_3d, flip_pairs):
    assert joints_3d.ndim in (2, 3)
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0] = -1 * joints[:, 0]
    # change left-right parts
    if flip_pairs is not None:
        print(joints)
        for pair in flip_pairs:
            print(pair)
            print(joints[pair[0]], joints[pair[1]])
            joints[pair[0]], joints[pair[1]] = joints[pair[1]], joints[pair[0]].copy()
    return joints

def flip_joints_3d(joints_3d, width, flip_pairs):
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    if flip_pairs is not None:
        for pair in flip_pairs:
            joints[pair[0], :, 0], joints[pair[1], :, 0] = \
                joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
            joints[pair[0], :, 1], joints[pair[1], :, 1] = \
                joints[pair[1], :, 1], joints[pair[0], :, 1].copy()
    joints[:, :, 0] *= joints[:, :, 1]
    return joints

class FlipAugmentation:
    def __init__(self, p, flip_pairs=None):
        self.p = p
        self.flip_pairs = flip_pairs
        
    def __call__(self, im, mask, obs):
        if random.random() <= self.p:
            im = flip_img(im.numpy())
            # mask = flip_img(mask)
            obs['objects'][0]['keypoints_2d'] = flip_joints_2d(np.array(obs['objects'][0]['keypoints_2d']), im.shape[1], self.flip_pairs)
            obs['camera']['K'][0,0] = - obs['camera']['K'][0,0]
            obs['camera']['K'][0,2] = im.shape[1] - 1 - obs['camera']['K'][0,2]
        return im, mask, obs
    
def rotate_joints_2d(joints_2d, width):    
    joints = joints_2d.copy()
    joints[:,  1] = joints_2d[:, 0]
    joints[:,  0] = width - joints_2d[:,  1] + 1 
    return joints

class RotationAugmentation:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, im, mask, obs):
        if random.random() <= self.p:
            h,w = im.shape[0],im.shape[1]
            im_copy = np.zeros((w,h,3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    im_copy[j][h-i-1]=im[i][j].astype(np.uint8)
            rgb = PIL.fromarray(im_copy)
            obs['objects'][0]['keypoints_2d'] = rotate_joints_2d(obs['objects'][0]['keypoints_2d'], im_copy.shape[1])
            kp3d = obs['objects'][0]['TCO_keypoints_3d']
            K = obs['camera']['K']
            # original_fx,original_fy,original_cx,original_cy = K[0][0],K[1][1],K[0][2],K[1][2]
            # 角度（弧度）表示旋转方向，顺时针旋转90度
            angle = np.pi / 2  # 90 degree
            K[0][2],K[1][2] = K[1][2],K[0][2]
            # set up rotation matrix
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            # new camera intrinsic matrix
            # obs['camera']['K'] = new_intrinsic_matrix
            for i in range(kp3d.shape[0]):
                kp3d[i] = np.dot(rotation_matrix, kp3d[i])
                   
            return rgb, mask, obs
        else:
            pass


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
    
def apply_color_jitter(rgb):
    color_factor=2*random.random()
    c_high = 1 + color_factor
    c_low = 1 - color_factor
    rgb=rgb.copy()
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * random.uniform(c_low, c_high), 0, 255)
    rgb = Image.fromarray(rgb)
    return rgb

def apply_occlusion(rgb):
    rgb=np.array(rgb)
    h,w,_ = rgb.shape
    synth_ymin, synth_h, synth_xmin, synth_w = occlusion_aug(None,np.array([h,w]), min_area=0.0, max_area=0.3, max_try_times=5)
    rgb = rgb.copy()
    rgb[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
    rgb = Image.fromarray(rgb)
    return rgb

def apply_rgb_aug(rgb):
    augSharpness = PillowSharpness(p=0.6, factor_interval=(0., 50.)) #0.3
    augContrast = PillowContrast(p=0.6, factor_interval=(0.7, 1.8)) #0.3
    augBrightness = PillowBrightness(p=0.6, factor_interval=(0.7, 1.8)) #0.3
    augColor = PillowColor(p=0.6, factor_interval=(0., 4.)) #0.3
    mask = None
    state = None
    rgb, mask, state = augSharpness(rgb, mask, state)
    rgb, mask, state = augContrast(rgb, mask, state)
    rgb, mask, state = augBrightness(rgb, mask, state)
    rgb, mask, state = augColor(rgb, mask, state)  
    rgb = np.array(rgb)
    rgb = Image.fromarray(rgb)
    return rgb