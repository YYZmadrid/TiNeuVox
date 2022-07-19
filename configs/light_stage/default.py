from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir = None,                
    human = None,
    dataset_type = 'light_stage',
    load2gpu_on_the_fly = True,  
    near = 2,
    far = 6,
    
    white_bkgd = False,     
    ratio = 0.5,
    num_render_views = 60,
    ndc = False,
    
    training_view = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    testing_view = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    render_train = False,
    render_test = False,
    
    begin_ith_frame = 0,
    frame_interval = 1,
    num_train_frame = 1,
    N_rand = 4096
)

''' Template of training options
'''
train_config = dict(
    N_iters=30000,                # number of optimization steps
    # N_rand=4096,                  # batch size (number of random rays per optimization step)
    lrate_feature=8e-2,           # lr of  voxel grid
    lrate_featurenet=8e-4,
    lrate_deformation_net=6e-4,
    lrate_densitynet=8e-4,
    lrate_timenet=8e-4,
    lrate_rgbnet=8e-4,           # lr of the mlp  
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    ray_sampler='in_maskcache',        # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.1,
    weight_rgbper=0.01,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=1e9,                   # count total variation before the given number of iterations
    tv_feature_before=10000,            # count total variation densely before the given number of iterations
    weight_tv_feature=0,
    pg_scale = [2000, 4000, 6000],
    skip_zero_grad_fields=['feature'],
)

''' Template of model and rendering options
'''

model_and_render = dict(
    num_voxels=160**3,          # expected number of voxel
    num_voxels_base=160**3,      # to rescale delta distance
    voxel_dim=6,                 # feature voxel grid dim
    voxel_channel=3,
    defor_depth=3,               # depth of the deformation MLP 
    net_width=256,             # width of the  MLP
    alpha_init=1e-3,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-4,           # threshold of alpha value to skip the fine stage sampled point
    stepsize=0.5,                 # sampling stepsize in volume rendering
    world_bound_scale=1.05,
    voxel_type='tineuvox',
)



del deepcopy
