import argparse
import copy
import os
import random
from selectors import EpollSelector
import time
from builtins import print
from tkinter import NO

import imageio
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from tqdm import tqdm, trange

from lib import tineuvox, utils
from lib.load_data import load_data
from torch.utils.tensorboard import SummaryWriter


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--eval_psnr", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--fre_test", type=int, default=200000,
                        help='frequency of test')
    parser.add_argument("--step_to_half", type=int, default=200000,
                        help='The iteration when fp32 becomes fp16')
    
    # gpus
    parser.add_argument("--gpus", type=int, nargs='+', default=[0])
    
    
    return parser

@torch.no_grad()
def render_viewpoints_light_stage(model, data_dict, render_kwargs, ndc, savedir=None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''

    rgbs = []
    depths = []
    accs = []
    dataset = data_dict['dataset']
    render_poses = dataset.render_poses
    for i, w2c in enumerate(tqdm(render_poses)):
        rgb, rays_o, rays_d, viewdirs, time_one, occ, mask_at_box, can_bounds = dataset.get_rays(split='test', index=i)  
        
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        time_one = time_one.flatten(0, -2)
        H, W, _ = rgb.shape
        mask_at_box = (mask_at_box.reshape(H, W) == 1).cpu().numpy()
        
        bacth_size=1000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, **render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0), viewdirs.split(bacth_size, 0), time_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }
        
        if render_kwargs['bg'] == 0:
            rgb = np.zeros((H, W, 3))
        else:
            rgb = np.ones((H, W, 3))
        depth = np.zeros((H, W))
        acc = np.zeros((H, W))
        rgb[mask_at_box] = render_result['rgb_marched'].cpu().numpy()
        depth[mask_at_box] = render_result['depth'].cpu().numpy()
        acc[mask_at_box] = render_result['alphainv_last'].cpu().numpy()
        
        rgbs.append(rgb)
        depths.append(depth)
        accs.append(acc)
        if i == 0:
            print('Testing', rgb.shape)

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    accs = np.array(accs)
    return rgbs, depths, accs


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)
    if cfg.data.dataset_type == 'light_stage':
        kept_keys = {
            'render_poses',
            'xyz_min',
            'xyz_max',
            'dataset',
            'near',
            'far',
        }
        for k in list(data_dict.keys()):
            if k not in kept_keys:
                data_dict.pop(k)
        return data_dict


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    writer = SummaryWriter(os.path.join(cfg.basedir, 'summaries', cfg.expname))
    
    # init model and optimizer
    start = 0
    
    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) :
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    voxel_type = model_kwargs.pop('voxel_type')
    if voxel_type == "mhe4d":
        frame_cfg = {
            'begin_ith_frame': cfg.data.begin_ith_frame,
            'num_train_frame': cfg.data.num_train_frame,
            'split_frame': cfg.data.split_frame,
        }
    else:
        frame_cfg = None
    model = tineuvox.TiNeuVox(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        voxel_type=voxel_type,
        frame_cfg=frame_cfg,
        **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
    }
    

    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # if global_step == args.step_to_half:
        #         model.feature.data=model.feature.data.half()
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, tineuvox.TiNeuVox):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        # random sample rays
        if cfg.data.dataset_type =='light_stage':
            target, rays_o, rays_d, viewdirs, times_sel, occ, mask_at_box, can_bounds = data_dict['dataset'].get_rays(split='train')
            mask_at_box = (mask_at_box == 1)
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            times_sel = times_sel.to(device)
            can_bounds = can_bounds.to(device)
        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, times_sel, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none = True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'][mask_at_box], target[mask_at_box])
        psnr = utils.mse2psnr(loss.detach())
        
        if cfg_train.weight_entropy_last > 0:
            pout = 1. - render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = F.mse_loss(pout[mask_at_box], occ[mask_at_box])
            loss += cfg_train.weight_entropy_last ** max(3000 - global_step, 0) * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        if global_step < cfg_train.tv_before and global_step > cfg_train.tv_after and global_step % cfg_train.tv_every==0:
            if cfg_train.weight_tv_feature > 0:
                model.feature_total_variation_add_grad(
                    cfg_train.weight_tv_feature / len(rays_o), global_step < cfg_train.tv_feature_before)
        optimizer.step()
        psnr_lst.append(psnr.item())
        
        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1 / decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step % args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction : iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/entropy_last_loss', entropy_last_loss.item(), global_step)
            writer.add_scalar('train/rgbper_loss', rgbper_loss.item(), global_step)
            writer.add_scalar('train/psnr', np.mean(psnr_lst), global_step)
            psnr_lst = []
                    
    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
        }, last_ckpt_path)
        print('scene_rep_reconstruction : saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict=None):

    # init
    print('train: start')
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    if cfg.data.dataset_type == 'light_stage':
        xyz_min, xyz_max = data_dict['xyz_min'], data_dict['xyz_max']

    # fine detail reconstruction
    eps_time = time.time()
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.model_and_render, cfg_train=cfg.train_config,
            xyz_min=xyz_min, xyz_max=xyz_max,
            data_dict=data_dict)
    eps_loop = time.time() - eps_time
    eps_time_str = f'{eps_loop//3600:02.0f}:{eps_loop//60%60:02.0f}:{eps_loop%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')

if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(args.gpus[0])
    seed_everything()
    data_dict = None
    
    # load images / poses / camera settings / data split
    data_dict = load_everything(args = args, cfg = cfg)

    # train
    if not args.render_only :
        train(args, cfg, data_dict = data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model_class = tineuvox.TiNeuVox
        model = utils.load_model(model_class, ckpt_path).to(device)
        near = data_dict['near']
        far = data_dict['far']
        stepsize = cfg.model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': near,
                'far': far,
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'render_depth': True,
            },
        }
    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok = True)
        raise NotImplementedError

        imageio.mimwrite(os.path.join(testsavedir, 'train_video.rgb.mp4'), utils.to8b(rgbs), fps = 30, quality = 8)
        imageio.mimwrite(os.path.join(testsavedir, 'train_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps = 30, quality = 8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        raise NotImplementedError

        imageio.mimwrite(os.path.join(testsavedir, 'test_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'test_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        if cfg.data.dataset_type  == 'light_stage':
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}_time')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, disps, accs = render_viewpoints_light_stage(data_dict=data_dict, savedir=testsavedir, ** render_viewpoints_kwargs) 
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality =8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.acc.mp4'), utils.to8b(accs / np.max(accs)), fps=30, quality =8)
        else:
            raise NotImplementedError

    print('Done')

