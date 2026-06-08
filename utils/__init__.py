'''
    Copyright (c) 2026 Chenxi Hu, Yichen Zhao, and Haiyang Chen, SJTU 2026.
    All rights reserved.

    This software is released under a custom Research-Only License.
    It is provided solely for academic research purpose related to the manuscript.

    Commercial use, clinical use, redistribution, sublicensing, or use in
    commercial product development is not permitted without prior written
    permission from the copyright holders.

    Citation:
    If you use this code or any part of this repository, please cite:
    [Citation information to be updated after publication]
'''

import torch
import numpy as np

def normalization(img):
    img_ = np.array(img).copy()
    len = img.shape[0]
    for i in range(len):
        tempImg = img_[i]
        pmax = np.percentile(tempImg, 99)
        pmin = np.percentile(tempImg, 1)
        tempImg[tempImg > pmax] = pmax
        tempImg[tempImg < pmin] = pmin
        tempImg = (tempImg - pmin) / (pmax - pmin)
        mean = np.mean(tempImg)
        std = np.std(tempImg)
        tempImg = (tempImg - mean) / std
        img_[i] = tempImg

    return torch.tensor(img_)

