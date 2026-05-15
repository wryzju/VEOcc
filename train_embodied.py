import pdb
import os, time, argparse, os.path as osp, numpy as np
import torch
import gc, copy
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torchrun --nproc_per_node=4 train_embodied.py
from utils.iou_as_iso import SSCMetrics, sync_ssc_metrics
from utils.loss_record import FunctionLossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_notddp

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import warnings

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


def set_only_fuser_train(model):
    model_for_call = model.module if hasattr(model, 'module') else model
    model_for_call.eval()
    if hasattr(model_for_call, 'local_fuser') and model_for_call.local_fuser is not None:
        model_for_call.local_fuser.train()


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
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

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='indoor_nyu', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    loss_func = my_model.loss

    if hasattr(my_model, 'depthanything') and my_model.depthanything is not None:
        my_model.depthanything.requires_grad_(False)

    # TODO
    # my_model.globalhead.requires_grad_(False)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
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

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', True)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # from loss import GPD_LOSS
    # loss_func = GPD_LOSS.build(cfg.loss).cuda()
    K_FRAMES = 30
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader) * max_num_epochs * K_FRAMES,
        lr_min=1e-6,
        warmup_t=500,  # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False)

    CalMeanIou = SSCMetrics(n_classes=12)
    CalMeanIou_Fov = SSCMetrics(n_classes=12)
    CalMeanIou_Global = SSCMetrics(n_classes=12)
    # resume and load
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if not distributed:
            state_dict = revise_ckpt_notddp(state_dict)
        else:
            state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))

    scenemeta_keys = [
        'global_scene_dim', 'global_scene_size', 'global_labels', 'global_pts', 'global_scene_origin', 'global_mask'
    ]
    metas_tensor_keys_inv = [
        'name',
        'cam2img',
        'world2img',
        'rgb_path',
        'depth_path',
        'num_depth',
        'occ_mask_valid',
        'img_shape',
        'img_aug_matrix',
    ]

    if is_main_process():
        my_writer = SummaryWriter(args.work_dir)
    # training
    while epoch < max_num_epochs:
        my_model.train()
        set_only_fuser_train(my_model)
        CalMeanIou_Global.reset()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = FunctionLossRecord(loss_func=loss_func)
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, labels) = data  # imgs [B, 1, K, 3, 480, 640]  labels [B, K, 60, 60, 36]
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

            # forward + backward + optimize
            data_time_e = time.time()

            my_model.module.scene_init(scenemetas)

            for i in range(K_Frames):
                img = imgs[:, :, i, :, :, :].unsqueeze(2)  # [B, 1, 1, 3, H, W]
                label = labels[:, i, :, :, :].unsqueeze(1)  # [B, 1, 60, 60, 36]
                meta = [monometa_list_cuda[i] for monometa_list_cuda in batch_monometa_list_cuda]

                with torch.cuda.amp.autocast(enabled=amp):
                    result_dict = my_model(imgs=img,
                                           metas=meta,
                                           scene_metas=scenemetas,
                                           points=None,
                                           label=label,
                                           grad_frames=cfg.grad_frames,
                                           test_mode=False)

                my_model.module.scene_update(result_dict, scenemetas, meta)
                loss, loss_dict = loss_func(result_dict, meta, label)
                loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step_update(global_iter)
                global_iter += 1

                if (i == K_Frames - 1):
                    scene_result_dicts = my_model.module.get_global_occ(scenemetas)

                    for scene_result_dict in scene_result_dicts:
                        global_valid_mask = scene_result_dict['mask']
                        global_label = scene_result_dict['label'][global_valid_mask].unsqueeze(0)
                        global_predict = scene_result_dict['predict'][global_valid_mask].unsqueeze(0)

                        global_predict[global_predict == 0] = 255
                        global_predict[global_predict == 12] = 0
                        global_label[global_label == 0] = 255
                        global_label[global_label == 12] = 0
                        global_predict = global_predict.cpu()
                        global_label = global_label.cpu()
                        CalMeanIou_Global.add_batch(global_predict, global_label)

                    my_model.module.scene_init(scenemetas)

                gc.collect()
                torch.cuda.empty_cache()

            valid_grad = True
            time_e = time.time()
            if not valid_grad and is_main_process():
                logger.info('[Nan Grad] Epoch %d Iter %5d' % (epoch + 1, i_iter))
                params, grads = [], []
                for name, param in my_model.named_parameters():
                    if param.requires_grad:
                        params.append(param.abs().mean().item())
                        grads.append(param.grad.abs().mean().item())
                logger.info('%.5f     %.5f     %.5f' %
                            (loss.item(), torch.mean(torch.tensor(params)).item(), torch.mean(torch.tensor(grads)).item()))

            if i_iter % print_freq == 0 and is_main_process():

                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                scene_names = [one_meta.get('scene_name', f'idx_{idx}') for idx, one_meta in enumerate(scenemetas)]
                logger.info('[TRAIN] scenes: ' + ', '.join(scene_names))
                logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch + 1, i_iter, len(train_dataset_loader)) + loss_info +
                            'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' %
                            (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))

                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()

            gc.collect()
            torch.cuda.empty_cache()

        sync_ssc_metrics(CalMeanIou_Global, device=torch.device(f'cuda:{gpu}'))
        global_status = CalMeanIou_Global.get_stats()
        global_sem_cls = global_status["iou_ssc"]
        global_sem = global_status["iou_ssc_mean"]
        global_geo = global_status["iou"]
        logger.info(f'Current global iou of sem is {global_sem_cls}')
        logger.info(f'Current global iou of sem is {global_sem}')
        logger.info(f'Current global iou of geo is {global_geo}')

        if is_main_process():
            my_writer.add_scalar('train/global_sem', global_sem, epoch)
            my_writer.add_scalar('train/global_geo', global_geo, epoch)

        # save checkpoint
        if is_main_process():
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_iou': best_val_iou,
                'best_val_miou': best_val_miou
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1

        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou.reset()
            CalMeanIou_Fov.reset()
            CalMeanIou_Global.reset()
            loss_record = FunctionLossRecord(loss_func=loss_func)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
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
                        with torch.cuda.amp.autocast(enabled=amp):
                            result_dict = my_model(imgs=img,
                                                   metas=meta,
                                                   scene_metas=scenemetas,
                                                   points=None,
                                                   label=label,
                                                   grad_frames=None,
                                                   test_mode=True)
                        my_model.module.scene_update(result_dict, scenemetas, meta)

                        loss, loss_dict = loss_func(result_dict, meta, label)
                        loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                        voxel_predict = result_dict['ce_input'].argmax(dim=1).long()  # [1, 60, 60, 36]
                        voxel_label = result_dict['ce_label'].long()  # [1, 60, 60, 36]

                        voxel_predict[voxel_predict == 0] = 255
                        voxel_predict[voxel_predict == 12] = 0
                        voxel_label[voxel_label == 0] = 255
                        voxel_label[voxel_label == 12] = 0
                        voxel_predict = voxel_predict.cpu()
                        voxel_label = voxel_label.cpu()
                        CalMeanIou.add_batch(voxel_predict, voxel_label)

                        voxel_predict = result_dict['ce_input'].argmax(dim=1).long()  # [1, 60, 60, 36]
                        voxel_label = result_dict['ce_label'].long()  # [1, 60, 60, 36]
                        this_fov_mask = torch.stack(
                            [one_meta['fov_mask'].to(device=voxel_predict.device, dtype=torch.bool) for one_meta in meta], dim=0)
                        voxel_predict = voxel_predict[this_fov_mask].unsqueeze(0)
                        voxel_label = voxel_label[this_fov_mask].unsqueeze(0)

                        voxel_predict[voxel_predict == 0] = 255
                        voxel_predict[voxel_predict == 12] = 0
                        voxel_label[voxel_label == 0] = 255
                        voxel_label[voxel_label == 12] = 0
                        voxel_predict = voxel_predict.cpu()
                        voxel_label = voxel_label.cpu()

                        CalMeanIou_Fov.add_batch(voxel_predict, voxel_label)

                        if (i == K_Frames - 1):
                            scene_result_dicts = my_model.module.get_global_occ(scenemetas)

                            for scene_result_dict in scene_result_dicts:
                                global_valid_mask = scene_result_dict['mask']
                                global_label = scene_result_dict['label'][global_valid_mask].unsqueeze(0)
                                global_predict = scene_result_dict['predict'][global_valid_mask].unsqueeze(0)

                                global_predict[global_predict == 0] = 255
                                global_predict[global_predict == 12] = 0
                                global_label[global_label == 0] = 255
                                global_label[global_label == 12] = 0
                                global_predict = global_predict.cpu()
                                global_label = global_label.cpu()

                                CalMeanIou_Global.add_batch(global_predict, global_label)

                            my_model.module.scene_init(scenemetas)

                    if i_iter_val % print_freq == 0 and is_main_process():
                        loss_info = loss_record.loss_info()
                        scene_names = [one_meta.get('scene_name', f'idx_{idx}') for idx, one_meta in enumerate(scenemetas)]
                        logger.info('[EVAL] scenes: ' + ', '.join(scene_names))
                        logger.info('[EVAL] Iter %5d/%d   ' % (i_iter_val, len(val_dataset_loader)) + loss_info)

                    gc.collect()
                    torch.cuda.empty_cache()

            sync_ssc_metrics(CalMeanIou, device=torch.device(f'cuda:{gpu}'))
            sync_ssc_metrics(CalMeanIou_Fov, device=torch.device(f'cuda:{gpu}'))
            sync_ssc_metrics(CalMeanIou_Global, device=torch.device(f'cuda:{gpu}'))

            global_status = CalMeanIou_Global.get_stats()
            global_sem_cls = global_status["iou_ssc"]
            global_sem = global_status["iou_ssc_mean"]
            global_geo = global_status["iou"]
            logger.info(f'Current global iou of sem is {global_sem_cls}')
            logger.info(f'Current global iou of sem is {global_sem}')
            logger.info(f'Current global iou of geo is {global_geo}')

            if is_main_process():
                my_writer.add_scalar('val/global_sem', global_sem, epoch)
                my_writer.add_scalar('val/global_geo', global_geo, epoch)

            stats = CalMeanIou.get_stats()
            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            logger.info(f'Current single val iou of sem_cls is {info_sem_cls}')
            logger.info(f'Current single val iou of sem is {info_sem}')
            logger.info(f'Current single val iou of geo is {info_geo}')

            stats_fov = CalMeanIou_Fov.get_stats()
            info_sem_cls_fov = stats_fov["iou_ssc"]
            info_sem_fov = stats_fov["iou_ssc_mean"]
            info_geo_fov = stats_fov["iou"]

            logger.info(f'Current fov val iou of sem_cls is {info_sem_cls_fov}')
            logger.info(f'Current fov val iou of sem is {info_sem_fov}')
            logger.info(f'Current fov val iou of geo is {info_geo_fov}')

            if is_main_process():
                my_writer.add_scalar('val/sem_fov', info_sem_fov, epoch)
                my_writer.add_scalar('val/geo_fov', info_geo_fov, epoch)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_embodied_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_embodied')
    parser.add_argument('--resume-from', type=str, default='')

    args, _ = parser.parse_known_args()
    main(args)
