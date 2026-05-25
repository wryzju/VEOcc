import pdb
import gc, os, time, argparse, os.path as osp, numpy as np

import torch
import torch.distributed as dist

from utils.loss_record import FunctionLossRecord, LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_notddp
from model.segmentor.utils import save_occ_points_label_ply, save_occ_label_ply

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import warnings

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    save_single_pred_dir = osp.join(args.work_dir, 'ply', 'single_pred')
    if args.save_ply:
        os.makedirs(save_pred_dir, exist_ok=True)
        os.makedirs(save_gt_dir, exist_ok=True)
        os.makedirs(save_single_pred_dir, exist_ok=True)

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

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = revise_ckpt(state_dict)
    my_model.load_state_dict(state_dict, strict=False)
    print('load ckpt successfully.')

    scenemeta_keys = [
        'global_scene_dim', 'global_scene_size', 'global_labels', 'global_pts', 'global_scene_origin', 'global_mask'
    ]
    metas_tensor_keys_inv = [
        'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path', 'num_depth', 'occ_mask_valid', 'img_shape', 'img_aug_matrix'
    ]

    # eval
    my_model.eval()
    loss_record = FunctionLossRecord(loss_func=loss_func)
    total_forward_time = 0.0
    num_forward = 0
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    flag_break = False
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, labels) = data  # imgs [1, 1, 30, 3, 480, 640]  labels [1, 30, 60, 60, 36]
            scenemetas = metas
            batch_monometa_list_cuda = []
            for scenemeta in scenemetas:
                for k, v in scenemeta.items():
                    if k in scenemeta_keys:
                        scenemeta[k] = torch.tensor(v).cuda()
                K_Frames = len(scenemeta['monometa_list'])
                monometa_list_cuda = []
                for i in range(K_Frames):
                    monometa = scenemeta['monometa_list'][i]
                    for k, v in monometa.items():
                        if not (k in metas_tensor_keys_inv):
                            monometa[k] = torch.tensor(v).cuda()
                    monometa_list_cuda.append(monometa)
                batch_monometa_list_cuda.append(monometa_list_cuda)

            my_model.module.scene_init(scenemetas)

            for i in range(K_Frames):
                img = imgs[:, :, i, :, :, :].unsqueeze(2)
                label = labels[:, i, :, :, :].unsqueeze(1)
                meta = [monometa_list_cuda[i] for monometa_list_cuda in batch_monometa_list_cuda]
                forward_start = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=amp):
                    result_dict = my_model(scenemeta=scenemeta,
                                           imgs=img,
                                           metas=meta,
                                           scene_metas=scenemetas,
                                           points=None,
                                           label=label,
                                           grad_frames=None,
                                           test_mode=True)
                my_model.module.scene_update(result_dict, scenemeta, meta)

                if num_forward >= num_warmup:
                    total_forward_time += time.perf_counter() - forward_start

                num_forward += 1
                if calc_fps and num_forward >= num_fpc_calc_forward + num_warmup:
                    flag_break = True
                    break

                loss, loss_dict = loss_func(result_dict, meta, label)
                loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                voxel_predict = result_dict['ce_input'].argmax(dim=1).long()  # [1, 60, 60, 36]

                if args.save_ply:
                    for batch_idx in range(voxel_predict.shape[0]):
                        frame_pred = voxel_predict[batch_idx:batch_idx + 1].clone()
                        frame_fov_mask = meta[batch_idx]['fov_mask']
                        if frame_fov_mask.dim() == 3:
                            frame_fov_mask = frame_fov_mask.unsqueeze(0)
                        frame_fov_mask = frame_fov_mask.to(device=frame_pred.device, dtype=torch.bool)

                        # Keep consistent with eval_mono save flow.
                        frame_pred[:, -1, :, :] = 12
                        frame_pred[:, :, -1, :] = 12
                        frame_pred[~frame_fov_mask] = 12

                        frame_pred = frame_pred.detach().cpu()
                        frame_origin = meta[batch_idx]['vox_origin']
                        curr_scene_name = scenemetas[batch_idx].get('scene_name', f'scene_{batch_idx:02d}')
                        frame_name = f'{int(i):05d}'
                        frame_curr_name = f'{curr_scene_name}/{frame_name}'
                        save_occ_label_ply(frame_pred, getattr(cfg, 'voxel_size', 0.08), frame_origin, save_single_pred_dir,
                                           frame_curr_name)

                scene_result_dicts = None
                if args.save_ply or (i == K_Frames - 1):
                    scene_result_dicts = my_model.module.get_global_occ(scenemetas)

                if args.save_ply:
                    for batch_idx, scene_result_dict in enumerate(scene_result_dicts):
                        global_valid_mask = scene_result_dict['mask'].to(torch.bool)
                        global_label = scene_result_dict['label'].long()
                        global_predict = scene_result_dict['predict'].long()
                        global_pts = scene_result_dict.get('global_pts', scenemetas[batch_idx]['global_pts'])

                        if isinstance(global_pts, torch.Tensor):
                            global_pts = global_pts.detach().cpu()

                        curr_scene_name = scene_result_dict.get('scene_name', f'scene_{batch_idx:02d}')
                        frame_name = f'{int(i):05d}'
                        pred_curr_name = f'{curr_scene_name}/{frame_name}'
                        gt_curr_name = f'{curr_scene_name}/{frame_name}'

                        save_pose_dir = osp.join(args.work_dir, 'ply', 'pose', curr_scene_name)
                        os.makedirs(save_pose_dir, exist_ok=True)
                        cam2img = meta[batch_idx]['cam2img']
                        cam2world = meta[batch_idx]['cam2world']
                        if isinstance(cam2img, torch.Tensor):
                            cam2img = cam2img.detach().cpu().numpy()
                        if isinstance(cam2world, torch.Tensor):
                            cam2world = cam2world.detach().cpu().numpy()

                        intrinsic_path = osp.join(save_pose_dir, f'intrinsic_{frame_name}.txt')
                        cam2world_path = osp.join(save_pose_dir, f'cam2world_{frame_name}.txt')
                        np.savetxt(intrinsic_path, cam2img[:3, :3], fmt='%.8f')
                        np.savetxt(cam2world_path, cam2world, fmt='%.8f')

                        save_occ_points_label_ply(global_pts[global_valid_mask], global_predict[global_valid_mask], save_pred_dir,
                                                  pred_curr_name)

                if (i == K_Frames - 1):
                    if scene_result_dicts is None:
                        scene_result_dicts = my_model.module.get_global_occ(scenemetas)

                    my_model.module.scene_init(scenemetas)

            if flag_break:
                break

            if i_iter_val % print_freq == 0 and is_main_process():
                loss_info = loss_record.loss_info()
                scene_names = [one_meta.get('scene_name', f'idx_{idx}') for idx, one_meta in enumerate(scenemetas)]
                logger.info('[EVAL] scenes: ' + ', '.join(scene_names))
                logger.info('[EVAL] Iter %5d/%d   ' % (i_iter_val, len(val_dataset_loader)) + loss_info)

            gc.collect()
            torch.cuda.empty_cache()

    if num_forward > 0 and calc_fps:
        avg_forward_time = total_forward_time / max(num_forward - num_warmup, 1)
        fps = 1.0 / avg_forward_time if avg_forward_time > 0 else float('inf')
        if with_depth_anything:
            logger.info(
                f'Average forward and update time per iteration with depth anything: {avg_forward_time:.6f} s, FPS: {fps:.3f}')
        else:
            logger.info(
                f'Average forward and update time per iteration without depth anything: {avg_forward_time:.6f} s, FPS: {fps:.3f}')
        return


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/eval_embodied_colmap.py')
    parser.add_argument('--work-dir', type=str, default='workdir/eval_embodied_colmap')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--calc-fps', action='store_true', help='whether to calculate fps')
    parser.add_argument('--save-ply',
                        action='store_true',
                        default=True,
                        help='save per-frame masked global pred/gt as ply under work_dir')

    args, _ = parser.parse_known_args()
    main(args)
