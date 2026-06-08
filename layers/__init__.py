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
import torch.nn as nn
import torch.nn.functional as nnf
device = torch.device("cuda:0")

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = torch.unsqueeze(grid, 2)
        grid = grid.type(torch.FloatTensor).to(device)
        grid = grid[:,[1,0],...]
        self.register_buffer('grid', grid)

    def forward(self, src, v):
        new_locs = self.grid + v
        shape = new_locs.shape
        integ = torch.zeros(src.shape)
        for i in range(0,shape[2]):
            new_locs[:, :, i, :, :] = 2 * new_locs[:, :, i, :, :] / (shape[3] - 1) - 1

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        for i in range(0, shape[2]):
            integ[:,:,i,:,:] = nnf.grid_sample(src[:,:,i,:,:],new_locs[:,i,:,:,:],align_corners=True,mode=self.mode)

        return integ.to(device)


class VecInt(nn.Module):

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'bilinear'

    def forward(self, V):
        [B,C,Nt,Nx,Ny] = V.shape
        OutV = torch.zeros((B,C,Nt,int(Nx*self.factor),int(Ny*self.factor))).to(device)
        if self.factor < 1:
            for i in range(Nt):
                OutV[:,:,i,:,:] = nnf.interpolate(V[:,:,i,:,:], align_corners=True, scale_factor=self.factor, mode=self.mode)
                OutV[:,:,i,:,:] = self.factor * OutV[:,:,i,:,:]

        elif self.factor > 1:
            for i in range(Nt):
                V[:, :, i, :, :] = self.factor * V[:, :, i, :, :]
                OutV[:,:,i,:,:] = nnf.interpolate(V[:,:,i,:,:], align_corners=True, scale_factor=self.factor, mode=self.mode)

        return OutV