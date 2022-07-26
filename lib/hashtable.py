import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import isprime


class MultiHashtable(nn.Module):
    def __init__(self, hashtable_cfg, xyz_min, xyz_max):
        """
        """
        super(MultiHashtable, self).__init__()

        mhe_cfg = dict(
            n_levels=16,
            n_entrys_per_level=2**19,
            base_resolution=16,
            f=2
        )
        mhe_cfg.update(hashtable_cfg)

        self.n_levels = mhe_cfg['n_levels']
        self.n_entrys_per_level = mhe_cfg['n_entrys_per_level']
        while True:
            if isprime(self.n_entrys_per_level):
                break
            else:
                self.n_entrys_per_level += 1

        self.b = 1.38
        self.base_resolution = mhe_cfg['base_resolution']
        self.f = mhe_cfg['f']
        self.ps = [1, 19349663, 83492791]

        self.data = nn.Parameter(
            torch.zeros((self.n_levels, self.n_entrys_per_level, self.f)))
        nn.init.kaiming_normal_(self.data)

        bbox = xyz_min.tolist() + xyz_max.tolist()
        self.bounds = torch.tensor(np.array(bbox).reshape((2, 3))).float().cuda()
        self.offsets = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], 
                                     [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]).float().cuda()

        self.entrys_size, self.entrys_num, = [], []
        for i in range(self.n_levels):
            grid_num = int((self.base_resolution * self.b**i) ** 3)
            grid_size = ((xyz_max - xyz_min).prod() / grid_num) ** (1 / 3)
            world_size = ((xyz_max - xyz_min) / grid_size).astype(np.int32)
            self.entrys_num.append(world_size)
            self.entrys_size.append((xyz_max - xyz_min) / (self.entrys_num[i] - 1))
        self.entrys_size = torch.tensor(self.entrys_size).float().cuda()
        self.entrys_num = torch.tensor(self.entrys_num).int().cuda()
        self.entrys_min = torch.zeros_like(self.entrys_num).int().cuda()
        
        self.start_hash = self.n_levels
        for i, n in enumerate(self.entrys_num):
            if n.prod() > self.data.shape[1]:
                self.start_hash = i
                break

    def forward(self, xyz):
        """
        xyz: n_points, 3
        """
        ind_xyz = xyz[None].repeat(self.n_levels, 1, 1)
        float_xyz = (ind_xyz - self.bounds[0][None, None]) / self.entrys_size[:, None, :] # [L, N, 3]
        int_xyz = (float_xyz[:, :, None] + self.offsets[None, None]).long() # [L, N, 8, 3]
        offset_xyz = float_xyz - int_xyz[:, :, 0] # [L, N, 3]  offset for each point in 3d grid
        int_xyz = torch.stack([
            torch.clamp(int_xyz[i], min=self.entrys_min[i], max=self.entrys_num[i] - 1)
            for i in range(self.n_levels)])
        # Only available under torch 1.10 with cuda 11.3
        # int_xyz = torch.clamp(int_xyz.transpose(0, -1), min=self.entrys_min, max=self.entrys_num-1).transpose(0, -1)
        ind = torch.zeros_like(int_xyz[..., 0]) # [L, N, 8]
        # __import__('ipdb').set_trace()

        sh = self.start_hash       
        ind[:sh] = int_xyz[:sh, ..., 0] * self.entrys_num[:sh, 1][:, None, None] * self.entrys_num[:sh, 2][:, None, None] + \
                   int_xyz[:sh, ..., 1] * self.entrys_num[:sh, 2][:, None, None] + \
                   int_xyz[:sh, ..., 2]
        nl = self.n_levels 
        ind[sh:nl] = torch.bitwise_xor(
            torch.bitwise_xor(int_xyz[sh:nl, ..., 0] * self.ps[0],
                              int_xyz[sh:nl, ..., 1] * self.ps[1]),
            int_xyz[sh:nl, ..., 2] * self.ps[2]) % self.n_entrys_per_level

        ind = ind.reshape(nl, -1)
        val = torch.gather(self.data, 1, ind[..., None].repeat(1, 1, self.f))
        val = val.reshape(nl, -1, 8, self.f)

        weights_xyz = torch.clamp(
            (1 - self.offsets[None, None]) + (2 * self.offsets[None, None] - 1.) * offset_xyz[:, :, None],
            min=0., max=1.)
        weights_xyz = weights_xyz[..., 0] * weights_xyz[..., 1] * weights_xyz[..., 2]

        val = (weights_xyz[..., None] * val).sum(dim=-2)
        val = val.permute(1, 0, 2).reshape(-1, nl * self.f)
        return val
