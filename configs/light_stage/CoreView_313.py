_base_ = './default.py'

expname = 'base/light_stage_313_20_frame_4dmhe'
basedir = './logs/light_stage'

data = dict(
    datadir = '/home/yanyunzhi/datasets/CoreView_313',
    human = 'CoreView_313',
    begin_ith_frame = 100,
    num_train_frame = 21,
    num_render_views = 100,
    split_frame = 10,
)

train_config = dict(
    N_iters = 20000,
    weight_entropy_last=0.01,
)

model_and_render = dict(
    mhe_cfg = dict(
        n_levels=16,
        n_entrys_per_level=2**19,
        base_resolution=16,
        f=2
    ),
    voxel_dim=2,
    voxel_channel=16,    
    voxel_type='mhe4d',
)