import tqdm
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
from mmengine import Config
from torch.utils.data import DataLoader, Subset


def main(args):

    cfg = Config.fromfile(args.py_config)
    distributed = True

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method=f"env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)

    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            # logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(my_model.cuda(), device_ids=[gpu], find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()

    # build dataloader
    from dataset import build_dataloader, custom_collate_fn
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    if distributed:

        def build_no_drop_loader(base_loader, batch_size, num_workers):
            base_dataset = base_loader.dataset
            total = len(base_dataset)
            rank_indices = list(range(rank, total, world_size))
            rank_subset = Subset(base_dataset, rank_indices)
            return DataLoader(
                dataset=rank_subset,
                batch_size=batch_size,
                collate_fn=custom_collate_fn,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )

        train_dataset_loader = build_no_drop_loader(
            train_dataset_loader,
            cfg.train_loader_config["batch_size"],
            cfg.train_loader_config["num_workers"],
        )
        val_dataset_loader = build_no_drop_loader(
            val_dataset_loader,
            cfg.val_loader_config["batch_size"],
            cfg.val_loader_config["num_workers"],
        )

    my_model.eval()

    metas_tensor_keys_inv = [
        'depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path', 'num_depth',
        'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix'
    ]

    dataloaders = [train_dataset_loader, val_dataset_loader]
    with torch.no_grad():
        for dataloader in dataloaders:
            if dataloader is train_dataset_loader:
                print('Processing train dataset...')
            else:
                print('Processing val dataset...')

            # for i_iter, data in enumerate(dataloader):
            for i_iter, data in enumerate(tqdm.tqdm(dataloader)):
                for i in range(len(data)):
                    if isinstance(data[i], torch.Tensor):
                        data[i] = data[i].cuda()
                (imgs, metas, label) = data
                for k, v in metas[0].items():
                    if not (k in metas_tensor_keys_inv):
                        metas[0][k] = torch.tensor(v).cuda()
                metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()

                result_dict = my_model(imgs=imgs,
                                       metas=metas,
                                       points=None,
                                       label=label,
                                       grad_frames=cfg.grad_frames,
                                       test_mode=False)
    return


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='./workdir/train_mono')
    parser.add_argument('--resume-from', type=str, default='')

    args, _ = parser.parse_known_args()
    main(args)
