
from __future__ import absolute_import, division, print_function

import os
import scipy.misc
import numpy as np
import PIL.Image as pil
import json

from kitti_utils import generate_depth_map
from cityscapes_utils import generate_depth_map_cityscapes
from .mono_dataset_v2 import mono_dataset

class cityscapes_dataset(mono_dataset):

    def __init__(self, *args, **kwargs):
        super(cityscapes_dataset, self).__init__(*args, **kwargs)
        self.K = np.array([[2262.52/2048, 0, 0.5, 0],
                           [0, 1096.98/1024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (2048, 1024)

    def check_depth(self):
        _, set_type, city, seq_index, frame_index = self.filenames[0].split()
        disp_filename = city + '_' + seq_index + '_' + frame_index + '_disparity' + self.img_ext
        disp_file_path = os.path.join(self.data_path, 'disparity_sequence', set_type, city,
                                     disp_filename)
        return os.path.isfile(disp_file_path)

    def get_color(self, side, set_type, city, seq_index, frame_index, do_flip):
        #crop = np.array([192, 1856, 256, 768])
        crop = (192, 256, 1856, 768)
        color = self.loader(self.get_image_path(side, set_type, city, seq_index, frame_index))
        color = color.crop(crop)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_image_path(self, side, set_type, city, seq_index, frame_index):
        if side == 'l':
            side_folder = 'leftImg8bit_sequence'
            side_file = 'leftImg8bit'
        else:
            side_folder = 'rightImg8bit_sequence'
            side_file = 'rightImg8bit'
        image_name = city + '_' + seq_index + '_' + frame_index + '_' + side_file + self.img_ext
        image_path = os.path.join(self.data_path, side_folder, set_type, city, image_name)
        return image_path

    def get_intrinsics(self, set_type, city, seq_index):
        cam_file_name = city + '_' + seq_index + '_000000_camera.json'
        calib_path = os.path.join(self.data_path, 'camera', set_type, city, cam_file_name)
        with open(calib_path) as calib_json:
            calib = json.load(calib_json)

        fx = calib['intrinsic']['fx']
        fy = calib['intrinsic']['fy']
        u0 = calib['intrinsic']['u0']
        v0 = calib['intrinsic']['v0']

        K = np.array([[fx, 0, u0, 0],
                      [0, fy, v0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K

    def get_depth(self, set_type, city, seq_index, frame_index, do_flip):
        cam_file_name = city + '_' + seq_index + '_000000_camera.json'
        disp_file_name = city + '_' + seq_index + '_' + frame_index + '_disparity.png'
        calib_path = os.path.join(self.data_path, 'camera', set_type, city, cam_file_name)
        disp_path = os.path.join(self.data_path, 'disparity_sequence', set_type, city, disp_file_name)

        depth_gt = generate_depth_map_cityscapes(calib_path, disp_path)

        return  depth_gt