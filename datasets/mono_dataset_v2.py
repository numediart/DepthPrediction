from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class mono_dataset(data.Dataset):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(mono_dataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # Need to specify augmentations differently in pytorch 1.0 compared with 0.4
        if int(torch.__version__.split('.')[0]) > 0:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        else:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        side, set_type, city, seq_index, frame_index = self.filenames[index].split()

        for i in self.frame_idxs:
            if i == 's':
                other_side = {'r': 'l', 'l': 'r'}[side]
                inputs[('color', i, -1)] = self.get_color(other_side, set_type, city, seq_index, frame_index, do_flip)
            else:
                f_i = int(frame_index) + i
                f_i_string = '{0:06d}'.format(f_i)
                inputs[('color', i, -1)] = self.get_color(side, set_type, city, seq_index, f_i_string, do_flip)

        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)] #because "to_tensor" in preprocess
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(set_type, city, seq_index, frame_index, do_flip)
            inputs['depth_gt'] = np.expand_dims(depth_gt, 0)
            inputs['depth_gt'] = torch.from_numpy(inputs['depth_gt'].astype(np.float32))

        if 's' in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == 'l' else 1
            stereo_T[0,3] = side_sign * baseline_sign * 0.1

            inputs['stereo_T'] = torch.from_numpy(stereo_T)
        return inputs

    def get_color(self, side, set_type, city, seq_index, frame_index, do_flip):
        raise NotImplementedError

    def get_intrinsics(self, set_type, city, seq_index):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, set_type, city, seq_index, frame_index, do_flip):
        raise NotImplementedError