_base_ = './default.py'

expname = 'base/light_stage_313_200_frame_4dmhe_nope'
basedir = './logs/light_stage'

data = dict(
    datadir = '/home/yanyunzhi/datasets/CoreView_313',
    human = 'CoreView_313',
    begin_ith_frame = 100,
    num_train_frame = 200,
    num_render_views = 200,
    split_frame = 200,
    N_rand = 2048, 
)

train_config = dict(
    lrate_decay = 30,
    N_iters = 60000,
    weight_entropy_last=0.01,
)

model_and_render = dict(
    mhe_cfg = dict(
        n_levels=16,
        n_entrys_per_level=2**21,
        base_resolution=16,
        f=2
    ),
    voxel_dim=2,
    voxel_channel=16,    
    voxel_type='mhe4d',
)