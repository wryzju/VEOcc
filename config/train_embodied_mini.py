batch_size = 2
ngpu = 4
optimizer_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-4 * batch_size * ngpu / 8,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)
grad_max_norm = 35
amp = False
seed = 1
print_freq = 1
eval_freq = 1
max_epochs = 2
load_from = 'checkpoints/mono_mini.pth'  # path/to/your/mono_checkpoint.pth
find_unused_parameters = True
track_running_stats = True

ignore_label = 0
empty_idx = 12  # 0 ignore, 1~11 objects, 12 empty
cls_dims = 13

voxel_size = 0.08  # 0.08m
scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
lss_downsample = [1, 1, 1]
grid_config = {
    'xbound': [0, scene_size[0], voxel_size * lss_downsample[0]],
    'ybound': [0, scene_size[1], voxel_size * lss_downsample[1]],
    'zbound': [0, scene_size[2], voxel_size * lss_downsample[2]],
    'dbound': [0.24, 6.00, 0.08],
}

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
scale_range = [0.01, 0.08]
image_size = [480, 640]
occ_size = [60, 60, 36]
resize_lim = [1.0, 1.0]
num_frames = 1
offset = 0
grad_frames = None

_dim_ = 96
num_cams = 1
norm_cfg = dict(type='GN', num_groups=_dim_ // 4, requires_grad=True)

loss = dict(type='MultiLoss',
            loss_cfgs=[
                dict(type='FocalLoss',
                     weight=100.0,
                     gamma=2.0,
                     alpha=0.25,
                     cls_freq=[
                         5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393,
                         30949122, 9871618, 3196722886
                     ],
                     ignore_label=ignore_label,
                     input_dict={
                         'pred': 'ce_input',
                         'target': 'ce_label',
                         'fov_mask': 'fov_mask'
                     }),
                dict(type='LovaszLoss',
                     weight=1.0,
                     ignore_label=ignore_label,
                     input_dict={
                         'lovasz_input': 'ce_input',
                         'lovasz_label': 'ce_label',
                         'fov_mask': 'fov_mask'
                     }),
                dict(type='Sem_Scal_Loss',
                     weight=1.0,
                     ignore_label=ignore_label,
                     sem_cls_range=[1, 12],
                     input_dict={
                         'pred': 'ce_input',
                         'ssc_target': 'ce_label',
                         'fov_mask': 'fov_mask'
                     }),
                dict(type='Geo_Scal_Loss',
                     weight=1.0,
                     empty_idx=empty_idx,
                     ignore_label=ignore_label,
                     input_dict={
                         'pred': 'ce_input',
                         'ssc_target': 'ce_label',
                         'fov_mask': 'fov_mask'
                     }),
            ])

model = dict(
    type='VoxelSegmentorOnline',
    build_depthanything=False,
    backbone=None,
    neck=dict(type='SECONDFPN',
              in_channels=[48, 80, 224, 2560],
              upsample_strides=[0.5, 1, 2, 4],
              out_channels=[_dim_, _dim_, _dim_, _dim_]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=_dim_ * 4,
        numC_Trans=_dim_,
        cam_channels=15,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=0.0,
    ),
    lifter=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=grid_config,
        data_config=dict(input_size=image_size),
    ),
    occ_encoder_backbone=dict(type='CustomResNet3D',
                              numC_input=_dim_,
                              num_layer=[2, 2, 2],
                              drop_path_rate=0.3,
                              num_channels=[_dim_, _dim_, _dim_],
                              stride=[1, 2, 2]),
    occ_encoder_neck=dict(type='GeneralizedLSSFPN',
                          in_channels=[_dim_, _dim_, _dim_],
                          out_channels=_dim_,
                          start_level=0,
                          num_outs=3,
                          norm_cfg=norm_cfg,
                          conv_cfg=dict(type='Conv3d'),
                          act_cfg=dict(type='ReLU', inplace=True),
                          upsample_cfg=dict(mode='trilinear', align_corners=False)),
    head=dict(
        type='OccHead',
        in_channels=[_dim_],
        out_channel=cls_dims,
        empty_idx=empty_idx,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=norm_cfg,
    ),
    enable_uvd_conf=True,
    local_fuser=dict(
        type='LocalLogitFusion',
        in_channels=_dim_,
        num_classes=cls_dims,
        loss_weight=1.0,
        ignore_label=ignore_label,
        class_frequencies=[
            5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393, 30949122, 9871618,
            3196722886
        ],
    ),
    loss_occ=loss,
)

data_path = 'data/occscannet'  # path/to/your/data/occscannet

train_dataset_config = dict(
    type='Scannet_Online_SceneOcc_Dataset',
    num_frames=num_frames,
    empty_idx=empty_idx,
    phase='train',
    data_tag='mini',  # 'mini' for mini-set
)

val_dataset_config = dict(
    type='Scannet_Online_SceneOcc_Dataset',
    num_frames=num_frames,
    empty_idx=empty_idx,
    phase='test',
    data_tag='mini',  # 'mini' for mini-set
)

train_wrapper_config = dict(
    type='Scannet_Scene_Occ_DatasetWrapper',
    final_dim=[480, 640],
    resize_lim=resize_lim,
    phase='train',
)

val_wrapper_config = dict(
    type='Scannet_Scene_Occ_DatasetWrapper',
    final_dim=[480, 640],
    resize_lim=resize_lim,
    phase='test',
)

train_loader_config = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

val_loader_config = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)
