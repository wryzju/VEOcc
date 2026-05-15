import timm
import torch
import numpy as np
import getpass
from copy import deepcopy
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
from loss import GPD_LOSS

import sys

from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from ...depthbranch.depthnet import DepthNet
from ...depthbranch.unet2d import DecoderBN
import torch.nn as nn
from PIL import Image
import cv2
import torch.nn.functional as F


@MODELS.register_module()
class VoxelSegmentor(BaseModule):

    def __init__(
        self,
        build_depthanything=False,
        depth_net=None,
        backbone=None,
        neck=None,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        lifter=None,
        encoder=None,
        head=None,
        init_cfg=None,
        loss_occ=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.use_offline_depth = not build_depthanything

        if build_depthanything:
            self.build_depthanything()
        else:
            self.depthanything = None

        assert backbone is None
        self.build_backbone()

        if neck is not None:
            self.neck = MODELS.build(neck)
        if depth_net is not None:
            self.depth_net = MODELS.build(depth_net)
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = MODELS.build(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = MODELS.build(occ_encoder_neck)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if head is not None:
            self.head = MODELS.build(head)
        if loss_occ is not None:
            self.loss_occ = GPD_LOSS.build(loss_occ)

    def build_backbone(self):
        basemodel_name = "tf_efficientnet_b7_ns"
        print("Loading base model ()...".format(basemodel_name), end="")
        # basemodel = torch.hub.load("rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True)
        username = getpass.getuser()
        basemodel = torch.hub.load(f"/home/{username}/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master",
                                   basemodel_name,
                                   pretrained=True,
                                   source="local")
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.backbone = basemodel

    def build_depthanything(self):
        # depth branch
        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384]
            },
            'vitb': {
                'encoder': 'vitb',
                'features': 128,
                'out_channels': [96, 192, 384, 768]
            },
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024]
            },
            'vitg': {
                'encoder': 'vitg',
                'features': 384,
                'out_channels': [1536, 1536, 1536, 1536]
            }
        }
        self.depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': 20})
        checkpoint = torch.load('./checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        self.depthanything.load_state_dict(new_state_dict)

    def extract_img_feat(self, imgs):
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/wyq/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)  # 1, 3, 480, 640

        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    this_x = vi(this_x)
                    feature_idx += 1
                    if feature_idx in [4, 5, 6, 8, 11]:
                        feature_x.append(this_x)
            else:
                this_x = v(this_x)
                feature_idx += 1
                if feature_idx in [4, 5, 6, 8, 11]:
                    feature_x.append(this_x)

        num_feat_levels = len(self.neck.in_channels)
        img_feats_backbone = feature_x[-num_feat_levels:]

        # list of [2560, 15, 20]
        img_feats_out = self.neck(img_feats_backbone)  # dict

        img_feats_reshaped = []
        for img_feat in img_feats_out:
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped[0].float()

    def prepare_camera_params(self, metas):

        B, N = len(metas), 1
        cam2world = torch.stack([meta['cam2world'] for meta in metas], dim=0).float()
        rot = cam2world[:, :3, :3].unsqueeze(1).repeat(1, N, 1, 1)
        tran = cam2world[:, :3, 3].unsqueeze(1).repeat(1, N, 1)

        intrin = torch.stack([meta['cam_k'] for meta in metas], dim=0).unsqueeze(1).repeat(1, N, 1, 1).float()
        if 'img_aug_matrix' in metas[0]:
            aug_matrix = torch.cat(
                [torch.tensor(meta['img_aug_matrix'], device=intrin.device, dtype=torch.float32) for meta in metas],
                dim=0,
            )
        else:
            aug_matrix = torch.eye(4).to(intrin.device, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        post_rot = aug_matrix[:, :, :3, :3]
        post_tran = aug_matrix[:, :, :3, 3]

        bda = None

        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)

        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)

        return [rot, tran, intrin, post_rot, post_tran, bda]

    def pred_depth(self, imgs, img_feats, metas, cam_params=None):
        if not hasattr(self, 'depth_net') or self.depth_net is None:
            if hasattr(self, 'depthanything') and self.depthanything is not None:
                self.depthanything.eval()
                depth_list = []
                for meta in metas:
                    image_ = meta['img_depthbranch']
                    depth = self.depthanything.infer_image(image_, 480, 640, 480)
                    depth_list.append(depth)
                depth = torch.stack(depth_list, dim=0)[:, None, ...]
            else:
                depth = torch.stack([metas[i]['depth_pred_np'] for i in range(len(metas))], dim=0)
                depth = depth[:, None, ...]

            # No depth branch: pass image features directly to the lifter.
            return depth, None, img_feats

        if hasattr(self, 'depthanything') and self.depthanything is not None:
            self.depthanything.eval()
            depth_list = []
            for meta in metas:
                image_ = meta['img_depthbranch']
                depth = self.depthanything.infer_image(image_, 480, 640, 480)
                depth_list.append(depth)
            depth = torch.stack(depth_list, dim=0)[:, None, ...]
        else:
            depth = torch.stack([metas[i]['depth_pred_np'] for i in range(len(metas))], dim=0)
            depth = depth[:, None, ...]

        if cam_params is None:
            cam_params = self.prepare_camera_params(metas)
        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        context, depth_distribution = self.depth_net([img_feats, mlp_input], depth)
        return depth, depth_distribution, context

    def obtain_voxel(self, context, depth, depth_distribution, metas, cam_params=None):
        if cam_params is None:
            cam_params = self.prepare_camera_params(metas)

        if depth_distribution is None:
            img_voxel_feature = self.lifter(context, cam_params, metas)
        else:
            img_voxel_feature = self.lifter(context, depth_distribution, cam_params, metas)

        return img_voxel_feature

    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)

        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)

        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        **kwargs,
    ):
        B, F, N, C, H, W = imgs.shape
        assert grad_frames is None
        assert F == 1, 'Only F=1 supported for now'
        output_dict = {}

        imgs = imgs.reshape(B * F, N, C, H, W)
        img_feats = self.extract_img_feat(imgs)
        cam_params = self.prepare_camera_params(metas)

        depth, depth_distribution, img_context_feature = self.pred_depth(imgs, img_feats, metas, cam_params=cam_params)
        output_dict['depth'] = depth
        output_dict['depth_distribution'] = depth_distribution

        img_voxel_feature = self.obtain_voxel(img_context_feature, depth, depth_distribution, metas, cam_params=cam_params)
        encoded_img_voxel_feature = self.occ_encoder(img_voxel_feature)
        occ_pred = self.head(encoded_img_voxel_feature)['output_voxels']
        output_dict['ce_input'] = occ_pred
        output_dict['ce_label'] = label.squeeze(1)
        output_dict['fov_mask'] = torch.stack([meta['fov_mask'] for meta in metas], dim=0)
        output_dict['img_voxel_feature'] = img_voxel_feature

        return output_dict

    def loss(self, output_dict, metas, label):
        total_loss = 0.
        loss_dict = {}

        if hasattr(self, 'depth_net') and self.depth_net is not None and self.depth_net.loss_depth_weight > 0:
            depth_gt = torch.stack([meta['depth_gt'] for meta in metas], dim=0)[:, None, ...]
            loss_depth = self.depth_net.get_depth_loss(depth_gt, output_dict['depth_distribution'])
            total_loss += loss_depth
            loss_dict['loss_depth'] = loss_depth.detach().item() / self.depth_net.loss_depth_weight

        total_loss_occ, loss_dict_occ = self.loss_occ(output_dict)

        total_loss += total_loss_occ
        loss_dict.update(loss_dict_occ)

        return total_loss, loss_dict
