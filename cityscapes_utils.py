from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter
from PIL import Image
import json


def generate_depth_map_cityscapes(calib_path, disp_path):
    crop = np.array([192, 1856, 256, 768])
    calib_json = open(calib_path)
    calib = json.load(calib_json)
    calib_json.close()
    baseline = calib['extrinsic']['baseline']
    fy = calib['intrinsic']['fy']

    img = Image.open(disp_path)
    disp = np.array(img)
    img.close()
    disp = disp[crop[2]:crop[3], crop[0]:crop[1]]
    disp = (disp - 1)/256.0

    depth = baseline * fy / disp
    depth[depth < 2] = 0
    depth[depth == np.Inf] = 0

    return depth
