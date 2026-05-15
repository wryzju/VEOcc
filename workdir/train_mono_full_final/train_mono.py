_dim_ = 96
amp = False
batch_size = 4
cls_dims = 13
data_path = 'data/occscannet'
empty_idx = 12
eval_freq = 1
find_unused_parameters = True
grad_frames = None
grad_max_norm = 35
grid_config = dict(
    dbound=[
        0.24,
        6.0,
        0.08,
    ],
    xbound=[
        0,
        4.8,
        0.08,
    ],
    ybound=[
        0,
        4.8,
        0.08,
    ],
    zbound=[
        0,
        2.88,
        0.08,
    ])
ignore_label = 0
image_size = [
    480,
    640,
]
load_from = None
loss = dict(
    loss_cfgs=[
        dict(
            alpha=0.25,
            cls_freq=[
                5080655412,
                722756,
                44793226,
                41084591,
                3416464,
                21897101,
                10609339,
                13846320,
                23470172,
                263393,
                30949122,
                9871618,
                3196722886,
            ],
            gamma=2.0,
            ignore_label=0,
            input_dict=dict(
                fov_mask='fov_mask', pred='ce_input', target='ce_label'),
            type='FocalLoss',
            weight=100.0),
        dict(
            ignore_label=0,
            input_dict=dict(
                fov_mask='fov_mask',
                lovasz_input='ce_input',
                lovasz_label='ce_label'),
            type='LovaszLoss',
            weight=1.0),
        dict(
            ignore_label=0,
            input_dict=dict(
                fov_mask='fov_mask', pred='ce_input', ssc_target='ce_label'),
            sem_cls_range=[
                1,
                12,
            ],
            type='Sem_Scal_Loss',
            weight=1.0),
        dict(
            empty_idx=12,
            ignore_label=0,
            input_dict=dict(
                fov_mask='fov_mask', pred='ce_input', ssc_target='ce_label'),
            type='Geo_Scal_Loss',
            weight=1.0),
    ],
    type='MultiLoss')
lss_downsample = [
    1,
    1,
    1,
]
max_epochs = 10
model = dict(
    backbone=None,
    build_depthanything=False,
    depth_net=dict(
        cam_channels=15,
        downsample=8,
        grid_config=dict(
            dbound=[
                0.24,
                6.0,
                0.08,
            ],
            xbound=[
                0,
                4.8,
                0.08,
            ],
            ybound=[
                0,
                4.8,
                0.08,
            ],
            zbound=[
                0,
                2.88,
                0.08,
            ]),
        loss_depth_type='kld',
        loss_depth_weight=0.0,
        numC_Trans=96,
        numC_input=384,
        type='GeometryDepth_Net'),
    head=dict(
        conv_cfg=dict(bias=False, type='Conv3d'),
        empty_idx=12,
        in_channels=[
            96,
        ],
        norm_cfg=dict(num_groups=24, requires_grad=True, type='GN'),
        num_level=1,
        occ_size=[
            60,
            60,
            36,
        ],
        out_channel=13,
        type='OccHead',
        with_cp=True),
    lifter=dict(
        data_config=dict(input_size=[
            480,
            640,
        ]),
        downsample=8,
        grid_config=dict(
            dbound=[
                0.24,
                6.0,
                0.08,
            ],
            xbound=[
                0,
                4.8,
                0.08,
            ],
            ybound=[
                0,
                4.8,
                0.08,
            ],
            zbound=[
                0,
                2.88,
                0.08,
            ]),
        type='LSSViewTransformer'),
    loss_occ=dict(
        loss_cfgs=[
            dict(
                alpha=0.25,
                cls_freq=[
                    5080655412,
                    722756,
                    44793226,
                    41084591,
                    3416464,
                    21897101,
                    10609339,
                    13846320,
                    23470172,
                    263393,
                    30949122,
                    9871618,
                    3196722886,
                ],
                gamma=2.0,
                ignore_label=0,
                input_dict=dict(
                    fov_mask='fov_mask', pred='ce_input', target='ce_label'),
                type='FocalLoss',
                weight=100.0),
            dict(
                ignore_label=0,
                input_dict=dict(
                    fov_mask='fov_mask',
                    lovasz_input='ce_input',
                    lovasz_label='ce_label'),
                type='LovaszLoss',
                weight=1.0),
            dict(
                ignore_label=0,
                input_dict=dict(
                    fov_mask='fov_mask',
                    pred='ce_input',
                    ssc_target='ce_label'),
                sem_cls_range=[
                    1,
                    12,
                ],
                type='Sem_Scal_Loss',
                weight=1.0),
            dict(
                empty_idx=12,
                ignore_label=0,
                input_dict=dict(
                    fov_mask='fov_mask',
                    pred='ce_input',
                    ssc_target='ce_label'),
                type='Geo_Scal_Loss',
                weight=1.0),
        ],
        type='MultiLoss'),
    neck=dict(
        in_channels=[
            48,
            80,
            224,
            2560,
        ],
        out_channels=[
            96,
            96,
            96,
            96,
        ],
        type='SECONDFPN',
        upsample_strides=[
            0.5,
            1,
            2,
            4,
        ]),
    occ_encoder_backbone=dict(
        drop_path_rate=0.3,
        numC_input=96,
        num_channels=[
            96,
            96,
            96,
        ],
        num_layer=[
            2,
            2,
            2,
        ],
        stride=[
            1,
            2,
            2,
        ],
        type='CustomResNet3D'),
    occ_encoder_neck=dict(
        act_cfg=dict(inplace=True, type='ReLU'),
        conv_cfg=dict(type='Conv3d'),
        in_channels=[
            96,
            96,
            96,
        ],
        norm_cfg=dict(num_groups=24, requires_grad=True, type='GN'),
        num_outs=3,
        out_channels=96,
        start_level=0,
        type='GeneralizedLSSFPN',
        upsample_cfg=dict(align_corners=False, mode='trilinear')),
    type='VoxelSegmentor')
ngpu = 4
norm_cfg = dict(num_groups=24, requires_grad=True, type='GN')
num_cams = 1
num_frames = 1
occ_size = [
    60,
    60,
    36,
]
offset = 0
optimizer_wrapper = dict(
    optimizer=dict(lr=0.0004, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
pc_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]
print_freq = 50
resize_lim = [
    1.0,
    1.0,
]
scale_range = [
    0.01,
    0.08,
]
scene_size = (
    4.8,
    4.8,
    2.88,
)
seed = 1
track_running_stats = True
train_dataset_config = dict(
    data_path='data/occscannet',
    data_tg='base',
    empty_idx=12,
    num_frames=1,
    offset=0,
    phase='train',
    type='Scannet_Scene_OpenOccupancy_Dataset_OffLine_Depth')
train_loader_config = dict(batch_size=4, num_workers=8, shuffle=True)
train_wrapper_config = dict(
    final_dim=[
        480,
        640,
    ],
    phase='train',
    resize_lim=[
        1.0,
        1.0,
    ],
    type='Scannet_Scene_Occ_DatasetWrapper')
val_dataset_config = dict(
    data_path='data/occscannet',
    data_tg='base',
    empty_idx=12,
    num_frames=1,
    offset=0,
    phase='test',
    type='Scannet_Scene_OpenOccupancy_Dataset_OffLine_Depth')
val_loader_config = dict(batch_size=4, num_workers=8, shuffle=False)
val_wrapper_config = dict(
    final_dim=[
        480,
        640,
    ],
    phase='test',
    resize_lim=[
        1.0,
        1.0,
    ],
    type='Scannet_Scene_Occ_DatasetWrapper')
voxel_size = 0.08
work_dir = 'workdir/train_mono_full_v1'
