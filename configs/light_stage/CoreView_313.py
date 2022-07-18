_base_ = './default.py'

expname = 'base/light_stage_313_200'
basedir = './logs/light_stage'

data = dict(
    datadir = '/home/yanyunzhi/datasets/CoreView_313',
    human = 'CoreView_313',
    begin_ith_frame = 100,
    num_train_frame = 200,
    num_render_views = 200
)

train_config = dict(
    N_iters = 30000,
)