import pdb
import gc, os, time, argparse, os.path as osp, numpy as np

import torch
import torch.distributed as dist

from utils.loss_record import FunctionLossRecord
from utils.load_save_util import revise_ckpt
from model.segmentor.utils import save_occ_label_ply

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging.logger import MMLogger
import warnings

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = False

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    print_freq = cfg.print_freq

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method=f"env://", world_size=world_size, rank=rank)

    # dist.barrier()
    torch.cuda.set_device(gpu)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    save_pred_dir = osp.join(args.work_dir, 'ply', 'pred')
    save_gt_dir = osp.join(args.work_dir, 'ply', 'gt')
    if args.save_ply:
        os.makedirs(save_pred_dir, exist_ok=True)
        os.makedirs(save_gt_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='indoor_nyu', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)

    calc_fps = args.calc_fps
    if calc_fps:
        assert cfg.batch_size == 1 and world_size == 1
        num_warmup = 10
        num_fpc_calc_forward = 100
        if my_model.depthanything is not None:
            with_depth_anything = True
        else:
            with_depth_anything = False
    else:
        num_warmup = 0
        num_fpc_calc_forward = 0

    loss_func = my_model.loss

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(my_model.cuda(), device_ids=[gpu], find_unused_parameters=find_unused_parameters)
        if hasattr(my_model, '_set_static_graph'):
            my_model._set_static_graph()
            logger.info('enabled DDP static graph.')
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # build dataloader
    from dataset import build_dataloader
    _, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    amp = cfg.get('amp', True)

    print('work dir: ', args.work_dir)

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = revise_ckpt(state_dict)
    my_model.load_state_dict(state_dict, strict=False)

    metas_tensor_keys_inv = [
        'depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path', 'num_depth',
        'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix'
    ]

    # eval
    my_model.eval()
    loss_record = FunctionLossRecord(loss_func=loss_func)
    total_forward_time = 0.0
    num_forward = 0
    sample_metric_records = []
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data

            for meta in metas:
                for k, v in meta.items():
                    if not (k in metas_tensor_keys_inv):
                        meta[k] = torch.tensor(v).cuda()

            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=amp):
                result_dict = my_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=None, test_mode=True)
            torch.cuda.synchronize()

            if num_forward >= num_warmup:
                total_forward_time += time.perf_counter() - forward_start
            num_forward += 1
            if calc_fps and num_forward >= num_fpc_calc_forward + num_warmup:
                break

            loss, loss_dict = loss_func(result_dict, metas, label)
            loss_record.update(loss=loss.item(), loss_dict=loss_dict)

            voxel_predict = result_dict['ce_input'].argmax(dim=1).long()  # [1, 60, 60, 36]

            if args.save_ply:
                # Keep only voxels in field-of-view for visualization output.
                fov_mask = result_dict['fov_mask']
                if fov_mask.dim() == 3:
                    fov_mask = fov_mask.unsqueeze(0)
                fov_mask = fov_mask.to(device=voxel_predict.device, dtype=torch.bool)
                save_voxel_predict = voxel_predict.clone()
                save_voxel_predict[:, -1, :, :] = 12
                save_voxel_predict[:, :, -1, :] = 12
                save_voxel_predict[~fov_mask] = 12

                voxel_origin = metas[0]['vox_origin']
                sample_name = metas[0]['name']
                save_occ_label_ply(save_voxel_predict.detach().cpu(), cfg.voxel_size, voxel_origin, save_pred_dir, sample_name)

            if i_iter_val % print_freq == 0 and is_main_process():
                loss_info = loss_record.loss_info()
                logger.info('[EVAL] Iter %5d/%d   ' % (i_iter_val, len(val_dataset_loader)) + loss_info)

            gc.collect()
            torch.cuda.empty_cache()

    if num_forward > 0 and calc_fps:
        avg_forward_time = total_forward_time / max(num_forward - num_warmup, 1)
        fps = 1.0 / avg_forward_time if avg_forward_time > 0 else float('inf')
        if with_depth_anything:
            logger.info(f'Average forward time per iteration with depth anything: {avg_forward_time:.6f} s, FPS: {fps:.3f}')
        else:
            logger.info(f'Average forward time per iteration without depth anything: {avg_forward_time:.6f} s, FPS: {fps:.3f}')

        return

    if args.record_sample_metrics and is_main_process():
        sample_metric_records.sort(key=lambda x: x['miou'], reverse=True)
        sample_metric_path = osp.join(args.work_dir, 'sample_iou_miou_rank.txt')
        with open(sample_metric_path, 'w') as f:
            f.write('rank\tname\tmiou\tiou\n')
            for rank_idx, record in enumerate(sample_metric_records, start=1):
                f.write(f"{rank_idx}\t{record['name']}\t{record['miou']:.6f}\t{record['iou']:.6f}\n")
        logger.info(f'Saved per-sample IoU/mIoU ranking to {sample_metric_path}')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/eval_mono_colmap.py')
    parser.add_argument('--work-dir', type=str, default='./workdir/eval_mono_colmap')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--calc-fps', action='store_true', help='whether to calculate fps')
    parser.add_argument('--save-ply',
                        action='store_true',
                        default=True,
                        help='save predicted and gt occupancy as ply under work_dir')
    parser.add_argument('--record-sample-metrics', action='store_true', help='record per-sample iou/miou ranking to work_dir')

    args, _ = parser.parse_known_args()
    main(args)
