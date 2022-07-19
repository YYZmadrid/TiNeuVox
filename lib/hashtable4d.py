import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import N, isprime
class MultiHashtable4d(nn.Module):
    def __init__(self, hashtable_cfg, xyz_min, xyz_max, time_cfg):
        """
        xyz_min, xyz_max is the same as MultiHashTable
        time_cfg: start frame, end_frame, total_frame
        """
    
        super(MultiHashtable4d, self).__init__()
        mhe_cfg = dict(
            n_levels=16,
            n_entrys_per_level=2**19,
            base_resolution=16,
            f=2
        )
        mhe_cfg.update(hashtable_cfg)
        
        self.total_frame = time_cfg['total_frame']
        self.start_frame = time_cfg['start_frame']
        self.end_frame = time_cfg['end_frame']

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
        self.ps = [1, 19349663, 83492791, 73856093]

        self.data = nn.Parameter(
            torch.zeros((self.n_levels, self.n_entrys_per_level, self.f)))
        nn.init.kaiming_normal_(self.data)

        bbox = xyz_min.tolist() + [self.start_frame / self.total_frame] \
                + xyz_max.tolist() + [self.end_frame / self.total_frame]
        self.bounds = torch.tensor(np.array(bbox).reshape((2, 4))).float().cuda()
        self.offsets = torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., 1., 1.], 
                                     [0., 1., 0., 0.], [0., 1., 0., 1.], [0., 1., 1., 0.], [0., 1., 1., 1.], 
                                     [1., 0., 0., 0.], [1., 0., 0., 1.], [1., 0., 1., 0.], [1., 0., 1., 1.], 
                                     [1., 1., 0., 0.], [1., 1., 0., 1.], [1., 1., 1., 0.], [1., 1., 1., 1.],
                                    ]).float().cuda()

        self.entrys_size, self.entrys_num, = [], []

        for i in range(self.n_levels):
            grid_num = int((self.base_resolution * self.b**i) ** 3)
            grid_size = ((xyz_max - xyz_min).prod() / grid_num) ** (1 / 3)
            world_size = ((xyz_max - xyz_min) / grid_size)
            xyzt_num = np.concatenate([world_size, \
                        np.array([self.end_frame-self.start_frame+1])]).astype(np.int32)
            self.entrys_num.append(xyzt_num)
            xyz_size = (xyz_max - xyz_min) / (world_size - 1)
            xyzt_size = np.concatenate([xyz_size, np.array([1./(self.total_frame-1)])])
            self.entrys_size.append(xyzt_size)
            
        self.entrys_size = torch.tensor(self.entrys_size).float().cuda()
        self.entrys_num = torch.tensor(self.entrys_num).int().cuda()
        self.entrys_min = torch.zeros_like(self.entrys_num).int().cuda()
        
        self.start_hash = self.n_levels
        for i, n in enumerate(self.entrys_num):
            if n.prod() > self.data.shape[1]:
                self.start_hash = i
                break

    def forward(self, xyz, t):
        """
        xyz: n_points, 3
        t: n_points, 1
        """
        
        # shift t to change its starting time
        t = t - self.start_frame / (self.total_frame - 1)
        assert torch.max(t).item() <= (self.end_frame) / (self.total_frame - 1)
        
        xyzt = torch.cat([xyz, t], axis=-1)
        
        ind_xyzt = xyzt[None].repeat(self.n_levels, 1, 1)
        float_xyzt = (ind_xyzt - self.bounds[0][None, None]) / self.entrys_size[:, None, :] # [L, N, 4]
        int_xyzt = (float_xyzt[:, :, None] + self.offsets[None, None]).long() # [L, N, 16, 4]
        offset_xyzt = float_xyzt - int_xyzt[:, :, 0] # [L, N, 4]  offset for each point in 4d grid
        int_xyzt = torch.stack([
            torch.clamp(int_xyzt[i], min=self.entrys_min[i], max=self.entrys_num[i] - 1)
            for i in range(self.n_levels)])
       
        ind = torch.zeros_like(int_xyzt[..., 0]) # [L, N, 16]
        sh = self.start_hash       
        ind[:sh] = int_xyzt[:sh, ..., 0] * self.entrys_num[:sh, 1][:, None, None] * \
        self.entrys_num[:sh, 2][:, None, None] * self.entrys_num[:sh, 3][:, None, None] + \
        int_xyzt[:sh, ..., 1] * self.entrys_num[:sh, 2][:, None, None] * self.entrys_num[:sh, 3][:, None, None] + \
        int_xyzt[:sh, ..., 2] * self.entrys_num[:sh, 3][:, None, None] + \
        int_xyzt[:sh, ..., 3]
        nl = self.n_levels 
        ind[sh:nl] = torch.bitwise_xor(
            torch.bitwise_xor(
            torch.bitwise_xor(int_xyzt[sh:nl, ..., 0] * self.ps[0],
                              int_xyzt[sh:nl, ..., 1] * self.ps[1]),
            int_xyzt[sh:nl, ..., 2] * self.ps[2]), 
            int_xyzt[sh:nl, ..., 3] * self.ps[3]) % self.n_entrys_per_level
        ind = ind.reshape(nl, -1)
        val = torch.gather(self.data, 1, ind[..., None].repeat(1, 1, self.f))
        val = val.reshape(nl, -1, 16, self.f)

        weights_xyzt = torch.clamp(
            (1 - self.offsets[None, None]) + (2 * self.offsets[None, None] - 1.) * offset_xyzt[:, :, None],
            min=0., max=1.)
        weights_xyzt = torch.prod(weights_xyzt, dim=-1)
        val = (weights_xyzt[..., None] * val).sum(dim=-2)
        val = val.permute(1, 0, 2).reshape(-1, nl * self.f)
        return val

class MHE4d(nn.Module):
    def __init__(self, hashtable_cfg, xyz_min, xyz_max, frame_cfg):
        super(MHE4d, self).__init__()
        self.begin_frame = frame_cfg['begin_ith_frame']
        self.num_train_frame = frame_cfg['num_train_frame']
        self.end_frame = self.begin_frame + self.num_train_frame - 1
        self.split_frame = frame_cfg['split_frame']
        for i in range(self.begin_frame, self.end_frame, self.split_frame):
            time_cfg = {
                'total_frame': self.num_train_frame,
                'start_frame': i-self.begin_frame,
                'end_frame': i-self.begin_frame+self.split_frame,
            }

            self.add_module('mhe_%d'%(i), MultiHashtable4d(hashtable_cfg, xyz_min, xyz_max, time_cfg))

    def forward(self, xyz, t):
        time = t[0]
        assert time >= 0. and time <= 1.
        frame_index =  time * (self.num_train_frame - 1) # interval
        mhe_index = (frame_index // self.split_frame) * self.split_frame + self.begin_frame
        if mhe_index <= self.begin_frame: 
            mhe_index = self.begin_frame
        elif mhe_index >= self.end_frame:
            mhe_index = self.end_frame - self.split_frame
        model = getattr(self, 'mhe_%d'%(mhe_index), 'mhe_%d'%(self.begin_frame))        
        feature = model(xyz, t)
        return feature
        