import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo
from .hashtable import MultiHashtable

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[],):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self._time, self._time_out = self.create_net()

    def create_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        return net_final(h)

    def forward(self, input_pts, ts):
        dx = self.query_time(input_pts, ts, self._time, self._time_out)
        input_pts_orig = input_pts[:, :3]
        out = input_pts_orig + dx
        return out

# Model
class RGBNet(nn.Module):
    def __init__(self, D=3, W=256, h_ch=256, views_ch=33, pts_ch=27, times_ch=17, output_ch=3):
        """ 
        """
        super(RGBNet, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = h_ch
        self.input_ch_views = views_ch
        self.input_ch_pts = pts_ch
        self.input_ch_times = times_ch
        self.output_ch = output_ch
        self.feature_linears = nn.Linear(self.input_ch, W)
        self.views_linears = nn.Sequential(nn.Linear(W+self.input_ch_views, W//2), nn.ReLU(), nn.Linear(W//2, self.output_ch))
        
    def forward(self, input_h, input_views):
        feature = self.feature_linears(input_h)
        feature_views = torch.cat([feature, input_views], dim=-1)
        outputs = self.views_linears(feature_views)
        return outputs

'''Model'''
class TiNeuVox(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0, add_cam=False,
                 alpha_init=None, fast_color_thres=0,
                 voxel_channel=3,
                 voxel_dim=0, defor_depth=3, net_width=128,
                 posbase_pe=10, viewbase_pe=4, timebase_pe=8, gridbase_pe=2,
                 voxel_type='tineuvox',
                 mhe_cfg=None,
                 **kwargs):
        
        super(TiNeuVox, self).__init__()
        

        self.add_cam = add_cam
        self.voxel_dim = voxel_dim
        self.defor_depth = defor_depth
        self.net_width = net_width
        self.posbase_pe = posbase_pe
        self.viewbase_pe = viewbase_pe
        self.timebase_pe = timebase_pe
        self.gridbase_pe = gridbase_pe
        times_ch = 2 * timebase_pe + 1
        views_ch = 3 + 3 * viewbase_pe * 2
        pts_ch = 3 + 3 * posbase_pe * 2
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
        self.voxel_channel = voxel_channel

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/ (1-alpha_init) - 1)
        print('TiNeuVox: set density bias shift to', self.act_shift)

        timenet_width = net_width
        timenet_depth = 1
        timenet_output = voxel_dim * (2 * gridbase_pe + 1)
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
        nn.Linear(timenet_width, timenet_output))
        if self.add_cam == True:
            views_ch = 3 + 3 * viewbase_pe * 2 + timenet_output
            self.camnet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
            nn.Linear(timenet_width, timenet_output))
            print('TiNeuVox: camnet', self.camnet)

        featurenet_width = net_width
        featurenet_depth = 1
        grid_dim = voxel_channel * voxel_dim * (2 * gridbase_pe + 1)
        input_dim = grid_dim + timenet_output + pts_ch
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )
        self.featurenet_width = featurenet_width
        self._set_grid_resolution(num_voxels)
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=3+3*posbase_pe*2, input_ch_time=timenet_output)
        input_dim = featurenet_width
        self.densitynet = nn.Linear(input_dim, 1)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('grid_poc', torch.FloatTensor([(2**i) for i in range(gridbase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('view_poc', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))

        self.voxel_dim = voxel_dim
        self.voxel_type = voxel_type
        
        if voxel_type == 'tineuvox':
            self.feature= torch.nn.Parameter(torch.zeros([1, self.voxel_dim, *self.world_size],dtype=torch.float32))
            self.mhe_cfg = None
        elif voxel_type == 'mhe':
            self.feature = MultiHashtable(hashtable_cfg=mhe_cfg, xyz_min=xyz_min, xyz_max=xyz_max)
            self.mhe_cfg = mhe_cfg
        else:
            raise NotImplementedError()
        
        self.rgbnet = RGBNet(W=net_width, h_ch=featurenet_width, views_ch=views_ch, pts_ch=pts_ch, times_ch=times_ch)
        
        if voxel_type == 'tineuvox':
            print('TiNeuVox: feature voxel grid', self.feature.shape)
        print('TiNeuVox: timenet mlp', self.timenet)
        print('TiNeuVox: deformation_net mlp', self.deformation_net)
        print('TiNeuVox: densitynet mlp', self.densitynet)
        print('TiNeuVox: featurenet mlp', self.featurenet)
        print('TiNeuVox: rgbnet mlp', self.rgbnet)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('TiNeuVox: voxel_size      ', self.voxel_size)
        print('TiNeuVox: world_size      ', self.world_size)
        print('TiNeuVox: voxel_size_base ', self.voxel_size_base)
        print('TiNeuVox: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'voxel_dim':self.voxel_dim,
            'defor_depth':self.defor_depth,
            'net_width':self.net_width,
            'posbase_pe':self.posbase_pe,
            'viewbase_pe':self.viewbase_pe,
            'timebase_pe':self.timebase_pe,
            'gridbase_pe':self.gridbase_pe,
            'add_cam': self.add_cam,
            'voxel_type': self.voxel_type,
            'voxel_channel': self.voxel_channel,
            'mhe_cfg': self.mhe_cfg
        }


    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        if self.voxel_type == 'tineuvox':
            print('TiNeuVox: scale_volume_grid start')
            ori_world_size = self.world_size
            self._set_grid_resolution(num_voxels)
            print('TiNeuVox: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
            self.feature = torch.nn.Parameter(
                F.interpolate(self.feature.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            pass
 
    def feature_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.feature.float(), self.feature.grad.float(), weight, weight, weight, dense_mode)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst


    def mult_dist_interp(self, ray_pts_delta):

        x_pad = math.ceil((self.feature.shape[2]-1)/4.0)*4-self.feature.shape[2]+1
        y_pad = math.ceil((self.feature.shape[3]-1)/4.0)*4-self.feature.shape[3]+1
        z_pad = math.ceil((self.feature.shape[4]-1)/4.0)*4-self.feature.shape[4]+1
        grid = F.pad(self.feature.float(),(0,z_pad,0,y_pad,0,x_pad))
        # three 
        vox_l = self.grid_sampler(ray_pts_delta, grid)
        vox_m = self.grid_sampler(ray_pts_delta, grid[:,:,::2,::2,::2])
        vox_s = self.grid_sampler(ray_pts_delta, grid[:,:,::4,::4,::4])
        vox_feature = torch.cat((vox_l,vox_m,vox_s), -1)

        if len(vox_feature.shape) == 1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        return vox_feature_flatten

    def set_ray_bounds(self, can_bounds):
        self.ray_min = can_bounds[0]
        self.ray_max = can_bounds[1]
        assert torch.all(self.ray_min > self.xyz_min)
        assert torch.all(self.ray_max < self.xyz_max)
        
    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.ray_min, self.ray_max, near, far, stepdist)        
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, mask_inbbox

    def forward(self, rays_o, rays_d, viewdirs, times_sel, cam_sel=None, bg_points_sel=None, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)
        times_emb = poc_fre(times_sel, self.time_poc)
        viewdirs_emb = poc_fre(viewdirs, self.view_poc)
        times_feature = self.timenet(times_emb)
        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox= self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)

        # pts deformation 
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id])
        # computer bg_points_delta
        if bg_points_sel is not None:
            bg_points_sel_emb = poc_fre(bg_points_sel, self.pos_poc)
            bg_points_sel_delta = self.deformation_net(bg_points_sel_emb, times_feature[:(bg_points_sel_emb.shape[0])])
            ret_dict.update({'bg_points_delta': bg_points_sel_delta})
        
        # voxel query interp
        if self.voxel_type == 'tineuvox':
            vox_feature_flatten = self.mult_dist_interp(ray_pts_delta)
        else:
            vox_feature_flatten = self.feature(ray_pts_delta)

        times_feature = times_feature[ray_id]
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        density_result = self.densitynet(h_feature)

        alpha = nn.Softplus()(density_result + self.act_shift)
        alpha = alpha.squeeze(-1)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            h_feature=h_feature[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            h_feature=h_feature[mask]

        viewdirs_emb_reshape = viewdirs_emb[ray_id]
        if self.add_cam == True:
            viewdirs_emb_reshape=torch.cat((viewdirs_emb_reshape, cams_feature[ray_id]), -1)
        rgb_logit = self.rgbnet(h_feature, viewdirs_emb_reshape)
        rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        with torch.no_grad():
            depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce='sum')
        ret_dict.update({'depth': depth})
        return ret_dict

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None



def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
