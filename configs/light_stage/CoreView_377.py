_base_ = './default.py'

expname = 'base/light_stage_377_200_tinevox'
basedir = './logs/light_stage'

data = dict(
    datadir = '/home/yanyunzhi/datasets/CoreView_377',
    human = 'CoreView_377',
    training_view = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    begin_ith_frame = 0,
    num_train_frame = 200,
    num_render_views = 200,
    # render_train = True,
)

train_config = dict(
    N_iters = 30000,
    N_rand = 1024,
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
    voxel_type='mhe',
)