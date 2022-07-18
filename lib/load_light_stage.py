from random import sample
import numpy as np
import os
import imageio
import cv2
import torch

class Dataset():
    def __init__(self, **data_dict):
        self.data_dict = data_dict
        self.data_root = data_dict['datadir']
        ann_file = os.path.join(self.data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']  

        novel_render_poses, novel_render_K, cams_R, cams_K = \
            render_poses(self.cams, data_dict['ratio'], data_dict['num_render_views'])
        training_view = data_dict['training_view']
        testing_view = data_dict['testing_view']
        if data_dict['render_train']:
            self.render_poses = cams_R[training_view]
            self.render_K = cams_K[training_view]
        elif data_dict['render_test']:
            self.render_poses = cams_R[testing_view]
            self.render_K = cams_K[testing_view]
        else:
            self.render_poses = novel_render_poses
            self.render_K = novel_render_K
            
        i = 0
        i = i + data_dict['begin_ith_frame']
        i_intv = data_dict['frame_interval']
        ni = data_dict['num_train_frame']

        self.num_train_frames = ni
        self.ims = np.array([
            np.array(ims_data['ims'])[training_view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[training_view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]]).ravel()
        
        if self.data_dict['human'] in ['CoreView_313', 'CoreView_315']:
            self.frame_inds = np.array([int(img_path.split('_')[4]) for img_path in self.ims])
        else:
            self.frame_inds = np.array([int(img_path[-10:-4]) for img_path in self.ims])

        self.times = (self.frame_inds - i) / (ni * i_intv)
        self.nrays = data_dict['N_rand']
        self.xyz_min, self.xyz_max = self.get_bounds(i, ni * i_intv)
        self.can_bounds = np.stack([self.xyz_min, self.xyz_max], axis=0)

    
    def get_mask(self, index):
        if self.data_dict['human'] in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(self.data_root, 'mask', self.ims[index])[:-4] + '.png'
        else:
            msk_path = os.path.join(self.data_root, 'mask_cihp', self.ims[index])[:-4] + '.png'

        msk_cihp = imageio.v2.imread(msk_path)
        msk = (msk_cihp != 0).astype(np.uint8)
        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def get_bounds(self, start, length):
        for i in range(start, start+length):
            vertices_path = os.path.join(self.data_root, 'new_vertices', '{}.npy'.format(i))
            xyz = np.load(vertices_path).astype(np.float32)
            
            if i == start:
                min_xyz = np.min(xyz, axis=0)
                max_xyz = np.max(xyz, axis=0)
            else:
                min_xyz_i = np.min(xyz, axis=0)
                max_xyz_i = np.max(xyz, axis=0)
                min_xyz = np.min(np.stack([min_xyz, min_xyz_i], axis=0), axis=0)
                max_xyz = np.max(np.stack([max_xyz, max_xyz_i], axis=0), axis=0)

        return min_xyz, max_xyz
    

    
    def get_rays(self, split=None, index=-1):        
        if split == 'train':
            index = np.random.randint(low=0, high=len(self.ims))
            img_path = os.path.join(self.data_root, self.ims[index])
            img = imageio.v2.imread(img_path).astype(np.float32) / 255.
            img = cv2.resize(img, (1024, 1024))
            msk = self.get_mask(index)
            cam_ind = self.cam_inds[index]
            K = np.array(self.cams['K'][cam_ind])
            D = np.array(self.cams['D'][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)

            R = np.array(self.cams['R'][cam_ind])
            T = np.array(self.cams['T'][cam_ind]) / 1000.

            # reduce the image resolution by ratio
            H, W = int(img.shape[0] * self.data_dict['ratio']), int(img.shape[1] * self.data_dict['ratio'])
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            img[msk == 0] = 0
            if self.data_dict['white_bkgd']:
                img[msk == 0] = 1
            K[:2] = K[:2] * self.data_dict['ratio']
            
            frame_index = self.frame_inds[index]
            can_bounds = self.get_bounds(frame_index, 1)
                    
            rgb, rays_o, rays_d, coord, mask_at_box = sample_ray_h36m(img, msk, K, R, T, can_bounds, self.nrays, split)
            occ = (msk[coord[:, 0], coord[:, 1]] == 1).astype(np.int32)
            viewdirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
            times = np.ones_like(rays_o)[..., :1] * self.times[index]
        else:
            pose = self.render_poses[index]
            K = self.render_K[index]
            R = pose[:3, :3]
            T = pose[:3, 3:]
            H, W = int(1024 * self.data_dict['ratio']), int(1024 * self.data_dict['ratio'])
            img, msk = np.zeros((H, W, 3)), np.zeros((H, W))
            frame_index = self.data_dict['begin_ith_frame'] + int(index * self.data_dict['frame_interval'] * self.num_train_frames / self.data_dict['num_render_views'] )
            can_bounds = self.get_bounds(frame_index, 2)
            rgb, rays_o, rays_d, coord, mask_at_box = sample_ray_h36m(img, msk, K, R, T, can_bounds, self.nrays, split)            
            occ = (msk[coord[:, 0], coord[:, 1]] == 1).astype(np.int32)
            viewdirs = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
            times = np.ones_like(rays_o)[..., :1] * index / self.data_dict['num_render_views'] 

        rgb = torch.Tensor(rgb)
        rays_o = torch.Tensor(rays_o)
        rays_d = torch.Tensor(rays_d)
        viewdirs = torch.Tensor(viewdirs)
        times = torch.Tensor(times)
        occ = torch.Tensor(occ)
        mask_at_box = torch.Tensor(mask_at_box)
        can_bounds = torch.Tensor(can_bounds)
        
        return rgb, rays_o, rays_d, viewdirs, times, occ, mask_at_box, can_bounds


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_2d_mask(bounds, K, pose, H, W):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    corners_3d = np.dot(corners_3d, pose[:, :3].T) + pose[:, 3].T
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box

def get_near_far_all(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
        
    if split == 'train':
        nsampled_rays = 0
        body_sample_ratio = 0.8
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        mask_at_box_list = []
        coord_list = []


        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body), n_body)]
            
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]
            coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            coord_list.append(coord[mask_at_box])

            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        mask_at_box = np.concatenate(mask_at_box_list)
        coord = np.concatenate(coord_list)

    else:
        rgb = img.astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far_all(bounds, ray_o, ray_d)
        ray_o, ray_d = ray_o[mask_at_box], ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)
        
    return rgb, ray_o, ray_d, coord, mask_at_box

def render_poses(cams, ratio, num_render_views):
    K = []
    RT = []
    lower_row = np.array([[0., 0., 0., 1.]])

    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        K[i][:2] = K[i][:2] * ratio

        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i]) / 1000.
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))
    
    RT = np.array(RT)
    camera_RT = RT.copy()
    camera_K = np.array(K.copy())
    
    RT[:] = np.linalg.inv(RT[:])
    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                        -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    def normalize(x):
        return x / np.linalg.norm(x)
    
    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec0_avg = up
        vec1 = normalize(np.cross(vec2, vec0_avg))
        vec0 = normalize(np.cross(vec1, vec2))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m


    def ptstocam(pts, c2w):
        tt = np.matmul(c2w[:3, :3].T, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
        return tt
    
    up = normalize(RT[:, :3, 0].sum(0))  
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    center = RT[:, :3, 3].mean(0)
    z_off = 1.3
    c2w = np.stack([up, vec1, vec2, center], 1)

    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, num_render_views + 1)[:-1]:
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        z = normalize(cam_pos_world - np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        mat = viewmatrix(z, up, cam_pos_world)
        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1], -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    render_K = np.repeat(np.array(K[0])[None], num_render_views, axis=0)
    
    return render_w2c, render_K, camera_RT, camera_K

def load_light_stage_data(**data_dict):
    dataset = Dataset(**data_dict)
    return dataset

if __name__ == '__main__':
    data_dict = dict(
        datadir='/home/yanyunzhi/datasets/CoreView_313',
        human='CoreView_313',
        dataset_type='light_stage',
        load2gpu_on_the_fly=True,
        
        white_bkgd = False,     
        ratio = 0.5,
        num_render_views = 60,
    
        training_view = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        begin_ith_frame = 0,
        frame_interval = 1,
        num_train_frame = 60,
        N_rand = 4096
    )
    
    dataset = Dataset(**data_dict)
    rgb, rays_o, rays_d, viewdirs, times = dataset.get_rays(split='train')
    
    __import__('ipdb').set_trace()
    