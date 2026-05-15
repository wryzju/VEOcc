import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from .voxel_segmentor import VoxelSegmentor


@MODELS.register_module()
class VoxelSegmentorOnline(VoxelSegmentor):

    def __init__(
        self,
        local_fuser=None,
        enable_uvd_conf=True,
        ignore_d_conf=False,  # for ablation
        ignore_uv_conf=False,  # for ablation
        fusion_strategy='weighted_sum_on_prob',
        colmap_mode=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for param in self.parameters():
            param.requires_grad = False

        self.enable_uvd_conf = enable_uvd_conf
        self.fusion_strategy = fusion_strategy
        self.valid_fusion_strategies = {
            'overwrite',
            'high_conf',
            'weighted_sum_on_logit',
            'weighted_sum_on_prob',
        }
        if self.fusion_strategy not in self.valid_fusion_strategies:
            raise ValueError(f'Invalid fusion_strategy={self.fusion_strategy}, expected one of {self.valid_fusion_strategies}')
        self.local_fuser = MODELS.build(local_fuser) if local_fuser is not None else None

        if self.enable_uvd_conf:
            self.uvd_depth_decay = nn.Parameter(torch.tensor(0.10, dtype=torch.float32), requires_grad=False)
            self.uvd_uv_decay = nn.Parameter(torch.tensor(1.50, dtype=torch.float32), requires_grad=False)
            self.uvd_min_conf = 0.01
            self.ignore_d_conf = ignore_d_conf
            self.ignore_uv_conf = ignore_uv_conf

        # dummy trainable param to avoid issues when the whole model is frozen
        if not any(param.requires_grad for param in self.parameters()):
            self.dummy_trainable_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.colmap_mode = colmap_mode

    def _build_zero_fuser_loss(self, reference_tensor):
        zero_loss = reference_tensor.sum() * 0.0
        grad_anchor = None
        if self.local_fuser is not None:
            for param in self.local_fuser.parameters():
                if param.requires_grad:
                    if grad_anchor is None:
                        grad_anchor = param.sum() * 0.0
                    else:
                        grad_anchor = grad_anchor + param.sum() * 0.0
        if grad_anchor is not None:
            zero_loss = zero_loss + grad_anchor
        return zero_loss

    def _compute_point_semantic_probs(self, coords_to_update, local_output, metas, batch_idx):
        metas_this = metas[batch_idx]

        def sample_volume(volume_logits, volume_feature, meta_ref, coords_ref, sample_mode='bilinear'):
            local_origin = meta_ref['vox_origin']
            local_scene_size = meta_ref['scene_size']

            size_x, size_y, size_z = volume_logits.shape[-3:]
            x_idx = (coords_ref[:, 0] - local_origin[0]) / local_scene_size[0] * (size_x - 1)
            y_idx = (coords_ref[:, 1] - local_origin[1]) / local_scene_size[1] * (size_y - 1)
            z_idx = (coords_ref[:, 2] - local_origin[2]) / local_scene_size[2] * (size_z - 1)

            valid = (x_idx >= 0) & (x_idx <= (size_x - 1)) & (y_idx >= 0) & (y_idx <= (size_y - 1)) & (z_idx >= 0) & (z_idx <= (
                size_z - 1))
            if valid.sum() == 0:
                return valid, None, None, None

            coords_valid = coords_ref[valid]
            x_idx = x_idx[valid]
            y_idx = y_idx[valid]
            z_idx = z_idx[valid]

            x_norm = 2.0 * x_idx / max(size_x - 1, 1) - 1.0
            y_norm = 2.0 * y_idx / max(size_y - 1, 1) - 1.0
            z_norm = 2.0 * z_idx / max(size_z - 1, 1) - 1.0
            sample_grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).view(1, -1, 1, 1, 3).float()

            volume_ncdhw = volume_logits.permute(0, 1, 4, 3, 2).contiguous()
            sampled_logits = F.grid_sample(
                volume_ncdhw,
                sample_grid,
                mode=sample_mode,
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(0).squeeze(-1).squeeze(-1)

            feat_ncdhw = volume_feature.permute(0, 1, 4, 3, 2).contiguous()
            sampled_feature = F.grid_sample(
                feat_ncdhw,
                sample_grid,
                mode=sample_mode,
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(0).squeeze(-1).squeeze(-1)

            return valid, coords_valid, sampled_logits, sampled_feature

        local_logits_this = local_output['ce_input'][batch_idx:batch_idx + 1]
        local_feature_this = local_output['img_voxel_feature'][batch_idx:batch_idx + 1]

        valid_this, coords_this, sampled_logits_this, sampled_feature_this = sample_volume(
            local_logits_this,
            local_feature_this,
            metas_this,
            coords_to_update,
        )
        if sampled_logits_this is None:
            return None, None, None, None, None

        if self.local_fuser is None or self.prev_local_pred is None or self.prev_metas is None:
            return coords_this, sampled_logits_this.softmax(dim=0), sampled_logits_this, None, None

        metas_prev = self.prev_metas[batch_idx]
        local_logits_prev = self.prev_local_pred['ce_input'][batch_idx:batch_idx + 1]
        local_feature_prev = self.prev_local_pred['img_voxel_feature'][batch_idx:batch_idx + 1]

        valid_prev, coords_prev, sampled_logits_prev, sampled_feature_prev = sample_volume(
            local_logits_prev,
            local_feature_prev,
            metas_prev,
            coords_this,
        )
        if sampled_logits_prev is None:
            return coords_this, sampled_logits_this.softmax(dim=0), sampled_logits_this, None, None

        local_fov_mask_prev = self.prev_local_pred['fov_mask'][batch_idx:batch_idx + 1]
        if local_fov_mask_prev.dim() == 4:
            local_fov_mask_prev = local_fov_mask_prev[:, None]
        local_fov_mask_prev = local_fov_mask_prev.float()

        fov_valid_prev, _, sampled_fov_prev, _ = sample_volume(
            local_fov_mask_prev,
            local_fov_mask_prev,
            metas_prev,
            coords_this,
            sample_mode='nearest',
        )

        fov_visible_prev = torch.zeros_like(valid_prev, dtype=torch.bool)
        if sampled_fov_prev is not None:
            fov_visible_prev[fov_valid_prev] = sampled_fov_prev.squeeze(0) > 0.5

        overlap_mask = valid_prev & fov_visible_prev
        if overlap_mask.sum() == 0:
            return coords_this, sampled_logits_this.softmax(dim=0), sampled_logits_this, None, None

        coords_overlap = coords_this[overlap_mask]
        sampled_logits_this_overlap = sampled_logits_this[:, overlap_mask]
        sampled_feature_this_overlap = sampled_feature_this[:, overlap_mask]
        overlap_mask_prev = overlap_mask[valid_prev]
        sampled_logits_prev_overlap = sampled_logits_prev[:, overlap_mask_prev]
        sampled_feature_prev_overlap = sampled_feature_prev[:, overlap_mask_prev]

        fuser_input_this = [sampled_logits_this_overlap, sampled_feature_this_overlap, coords_overlap, metas_this]
        fuser_input_prev = [sampled_logits_prev_overlap, sampled_feature_prev_overlap, coords_overlap, metas_prev]
        fused_overlap_logits = self.local_fuser.fuse(fuser_input_this, fuser_input_prev)

        if fused_overlap_logits is None:
            return coords_this, sampled_logits_this.softmax(dim=0), sampled_logits_this, None, None

        fused_logits_this = sampled_logits_this.clone()
        fused_logits_this[:, overlap_mask] = fused_overlap_logits

        return coords_this, fused_logits_this.softmax(dim=0), fused_logits_this, coords_overlap, fused_overlap_logits

    def _build_scene_update_payload(self, local_output, metas, scene_metas, image_hw=None):
        scene_update_payload = []
        local_semantic_filed = local_output['ce_input']
        local_semantic_filed = local_semantic_filed.float()  # [B, C, X, Y, Z]

        batch_size = local_semantic_filed.shape[0]

        for batch_idx in range(batch_size):

            # For convienience, we directly use the global mask from this frame as the mask to select points for fusion.
            # During the robot's exploration process, the coordinates of points can be determined based on the camera pose and fov.
            if not self.colmap_mode:
                global_mask_from_thisframe = metas[batch_idx]['mask_in_global_from_this'].bool()
                global_xyz = scene_metas[batch_idx]['global_pts']
                coords_to_update = global_xyz[global_mask_from_thisframe]  # [N, 3]
            else:
                local_vox_xyz = metas[batch_idx]['occ_xyz'].view(-1, 3)  # [60*60*36, 3]
                fov_mask = metas[batch_idx]['fov_mask'].view(-1).bool()

                # filter by FOV
                coords_to_update = local_vox_xyz[fov_mask].to(device=local_semantic_filed.device)

                if coords_to_update.shape[0] == 0:
                    scene_update_payload.append({
                        'coords_to_update': None,
                        'semantic_probs': None,
                        'semantic_logits': None,
                        'confidences': None,
                        'fused_coords': None,
                        'fused_logits': None,
                    })
                    continue

            payload_this = {
                'coords_to_update': None,
                'semantic_probs': None,
                'semantic_logits': None,
                'confidences': None,
                'fused_coords': None,
                'fused_logits': None,
            }
            if not self.colmap_mode and global_mask_from_thisframe.sum() == 0:
                scene_update_payload.append(payload_this)
                continue

            # semantics
            coords_to_update, semantic_probs, semantic_logits, fused_coords, fused_logits = self._compute_point_semantic_probs(
                coords_to_update,
                local_output,
                metas,
                batch_idx,
            )
            if semantic_probs is None:
                scene_update_payload.append(payload_this)
                continue

            # confs
            pred_confidence = None
            confidences = self._compute_point_confidence(
                sampled_probs=semantic_probs,
                pred_confidence=pred_confidence,
                sample_xyz_world=coords_to_update,
                meta=metas[batch_idx],
                image_hw=image_hw,
            )
            payload_this['coords_to_update'] = coords_to_update
            payload_this['semantic_probs'] = semantic_probs.permute(1, 0)
            payload_this['semantic_logits'] = semantic_logits.permute(1, 0) if semantic_logits is not None else None
            payload_this['confidences'] = confidences
            payload_this['fused_coords'] = fused_coords
            payload_this['fused_logits'] = fused_logits.permute(1, 0) if fused_logits is not None else None

            scene_update_payload.append(payload_this)

        self.prev_local_pred = {
            'img_voxel_feature': local_output['img_voxel_feature'].detach().clone(),
            'ce_input': local_output['ce_input'].detach().clone(),
            'fov_mask': local_output['fov_mask'].detach().clone(),
        }
        self.prev_metas = [{
            'vox_origin':
            meta['vox_origin'].detach().cpu().clone()
            if torch.is_tensor(meta['vox_origin']) else copy.deepcopy(meta['vox_origin']),
            'scene_size':
            meta['scene_size'].detach().cpu().clone()
            if torch.is_tensor(meta['scene_size']) else copy.deepcopy(meta['scene_size']),
        } for meta in metas]

        return scene_update_payload

    def _compute_point_confidence(self, sampled_probs, pred_confidence=None, sample_xyz_world=None, meta=None, image_hw=None):
        num_points = sampled_probs.shape[-1]
        confidence = torch.ones(num_points, device=sampled_probs.device, dtype=sampled_probs.dtype)

        if self.enable_uvd_conf:
            uvd_confidence = self._compute_uvd_confidence(
                sample_xyz_world=sample_xyz_world,
                meta=meta,
                image_hw=image_hw,
            )
            confidence = confidence * uvd_confidence.to(device=confidence.device, dtype=confidence.dtype)

        return confidence

    def _compute_uvd_confidence(self, sample_xyz_world, meta, image_hw):
        if sample_xyz_world is None or meta is None or image_hw is None:
            return torch.ones(0, device=self.visible_conf_weight.device, dtype=self.visible_conf_weight.dtype)

        device = sample_xyz_world.device
        dtype = sample_xyz_world.dtype
        num_points = sample_xyz_world.shape[0]
        if num_points == 0:
            return torch.ones(0, device=device, dtype=dtype)

        height, width = image_hw
        height = float(height)
        width = float(width)

        cam2world = meta['cam2world']
        if not torch.is_tensor(cam2world):
            cam2world = torch.tensor(cam2world, device=device, dtype=dtype)
        else:
            cam2world = cam2world.to(device=device, dtype=dtype)

        cam_k = meta['cam_k']
        if not torch.is_tensor(cam_k):
            cam_k = torch.tensor(cam_k, device=device, dtype=dtype)
        else:
            cam_k = cam_k.to(device=device, dtype=dtype)
        if cam_k.shape[0] == 4 and cam_k.shape[1] == 4:
            cam_k = cam_k[:3, :3]

        cam_pos = cam2world[:3, 3]
        rot = cam2world[:3, :3]
        points_cam = (sample_xyz_world - cam_pos.unsqueeze(0)) @ rot

        z_raw = points_cam[:, 2]
        z = z_raw.clamp(min=1e-4)
        fx = cam_k[0, 0]
        fy = cam_k[1, 1]
        cx = cam_k[0, 2]
        cy = cam_k[1, 2]
        u = fx * (points_cam[:, 0] / z) + cx
        v = fy * (points_cam[:, 1] / z) + cy

        depth_decay = F.softplus(self.uvd_depth_decay).to(device=device, dtype=dtype)
        uv_decay = F.softplus(self.uvd_uv_decay).to(device=device, dtype=dtype)

        depth_conf = torch.exp(-depth_decay * z)

        dist_left = u
        dist_right = (width - 1.0) - u
        dist_top = v
        dist_bottom = (height - 1.0) - v
        dist_to_border = torch.min(torch.min(dist_left, dist_right), torch.min(dist_top, dist_bottom))
        max_dist = max(min(width, height) * 0.5, 1.0)
        dist_to_border_norm = torch.clamp(dist_to_border / max_dist, min=0.0, max=1.0)
        border_proximity = 1.0 - dist_to_border_norm
        uv_conf = torch.exp(-uv_decay * border_proximity)

        in_front_mask = z_raw > 1e-4
        in_frame_mask = (u >= 0) & (u <= (width - 1.0)) & (v >= 0) & (v <= (height - 1.0))

        if self.ignore_d_conf:
            depth_conf = torch.ones_like(depth_conf)
        if self.ignore_uv_conf:
            uv_conf = torch.ones_like(uv_conf)

        final_conf = depth_conf * uv_conf
        min_conf = torch.tensor(self.uvd_min_conf, device=device, dtype=dtype)
        final_conf = torch.clamp(final_conf, min=min_conf, max=1.0)
        final_conf = torch.where(in_front_mask & in_frame_mask, final_conf, min_conf)
        return final_conf

    def scene_init(self, scenemetas):
        self.scene_names = []
        self.global_scene_origins = []
        self.global_semantics = []
        self.global_coords = []
        self.global_confidences = []
        self.global_logits = []
        self.global_fusion_counts = []
        self.K_frams = 0
        self.voxel_size = 0.08  # meter

        self.prev_local_pred = None
        self.prev_metas = None

        # for computing loss
        self.global_labels = []
        self.global_xyzs = []

        # for eval
        self.global_masks_thistime = []

        for scenemeta in scenemetas:
            self.scene_names.append(scenemeta['scene_name'])
            self.global_scene_origins.append(scenemeta['global_scene_origin'])
            self.K_frams = len(scenemeta['valid_img_paths']) if self.K_frams == 0 else self.K_frams
            assert self.K_frams == len(
                scenemeta['valid_img_paths']), 'All scenes should have the same K_Frames for multi batch processing.'
            self.global_semantics.append(None)
            self.global_coords.append(None)
            self.global_confidences.append(None)
            self.global_logits.append(None)
            self.global_fusion_counts.append(None)

            self.global_labels.append(scenemeta['global_labels'])  # [x_dim, y_dim, z_dim]
            self.global_xyzs.append(scenemeta['global_pts'])

            self.global_masks_thistime.append(torch.zeros_like(scenemeta['global_mask']).bool())

    def scene_update(self, output_dict, scenemeta, thismeta):
        scene_update_payload = output_dict.get('scene_update_payload', None)
        if scene_update_payload is None:
            return

        batch_size = len(scene_update_payload)

        for batch_idx in range(batch_size):
            payload_this = scene_update_payload[batch_idx]

            # for eval: always record visibility from this frame
            if not self.colmap_mode:
                global_mask_from_thisframe = thismeta[batch_idx]['mask_in_global_from_this'].bool()
                self.global_masks_thistime[batch_idx] = self.global_masks_thistime[batch_idx] | global_mask_from_thisframe

            coords_to_update = payload_this['coords_to_update']
            semantic_probs = payload_this['semantic_probs']
            semantic_logits = payload_this.get('semantic_logits', None)
            confidences = payload_this['confidences']

            if coords_to_update is None or semantic_probs is None or confidences is None:
                # Nothing valid to fuse for this frame/batch item.
                continue

            semantic_probs = semantic_probs.detach().clone()
            semantic_logits = semantic_logits.detach().clone() if semantic_logits is not None else None

            if coords_to_update is not None:
                global_origin = self.global_scene_origins[batch_idx]
                voxel_size = self.voxel_size
                gx = ((coords_to_update[:, 0] - global_origin[0]) / voxel_size).long()
                gy = ((coords_to_update[:, 1] - global_origin[1]) / voxel_size).long()
                gz = ((coords_to_update[:, 2] - global_origin[2]) / voxel_size).long()
                current_coords = torch.stack([gx, gy, gz], dim=1)

                P1, P2 = 73856093, 19349663
                current_hash = gx * P1 + gy * P2 + gz  # [N]

                if self.global_coords[batch_idx] is None:
                    self.global_coords[batch_idx] = current_coords
                    self.global_semantics[batch_idx] = semantic_probs
                    self.global_confidences[batch_idx] = confidences
                    self.global_logits[batch_idx] = semantic_logits
                else:
                    old_coords = self.global_coords[batch_idx]
                    old_sem = self.global_semantics[batch_idx]
                    old_conf = self.global_confidences[batch_idx]
                    old_logits = self.global_logits[batch_idx]

                    # 保证 device 一致（很重要）
                    device = semantic_probs.device
                    old_coords = old_coords.to(device)
                    old_sem = old_sem.to(device)
                    old_conf = old_conf.to(device)
                    if old_logits is not None:
                        old_logits = old_logits.to(device)
                    if semantic_logits is not None:
                        semantic_logits = semantic_logits.to(device)

                    P1, P2 = 73856093, 19349663
                    old_hash = old_coords[:, 0] * P1 + old_coords[:, 1] * P2 + old_coords[:, 2]
                    current_hash = gx * P1 + gy * P2 + gz
                    sorted_old_hash, old_perm = torch.sort(old_hash)
                    sorted_new_hash, new_perm = torch.sort(current_hash)
                    unique_hash = torch.unique(torch.cat([old_hash, current_hash], dim=0))

                    if sorted_old_hash.numel() == 0:
                        old_pos = torch.zeros_like(unique_hash, dtype=torch.long)
                        old_valid = torch.zeros_like(unique_hash, dtype=torch.bool)
                    else:
                        old_pos = torch.searchsorted(sorted_old_hash, unique_hash)
                        old_pos_safe = torch.clamp(old_pos, max=sorted_old_hash.shape[0] - 1)
                        old_valid = (old_pos < sorted_old_hash.shape[0]) & (sorted_old_hash[old_pos_safe] == unique_hash)
                        old_pos = old_pos_safe

                    if sorted_new_hash.numel() == 0:
                        new_pos = torch.zeros_like(unique_hash, dtype=torch.long)
                        new_valid = torch.zeros_like(unique_hash, dtype=torch.bool)
                    else:
                        new_pos = torch.searchsorted(sorted_new_hash, unique_hash)
                        new_pos_safe = torch.clamp(new_pos, max=sorted_new_hash.shape[0] - 1)
                        new_valid = (new_pos < sorted_new_hash.shape[0]) & (sorted_new_hash[new_pos_safe] == unique_hash)
                        new_pos = new_pos_safe

                    num_voxels = unique_hash.shape[0]
                    num_classes = old_sem.shape[1]

                    old_coords_u = torch.zeros((num_voxels, 3), device=device, dtype=old_coords.dtype)
                    new_coords_u = torch.zeros((num_voxels, 3), device=device, dtype=old_coords.dtype)
                    old_sem_u = torch.zeros((num_voxels, num_classes), device=device, dtype=old_sem.dtype)
                    new_sem_u = torch.zeros((num_voxels, num_classes), device=device, dtype=old_sem.dtype)
                    old_conf_u = torch.zeros((num_voxels, ), device=device, dtype=old_conf.dtype)
                    new_conf_u = torch.zeros((num_voxels, ), device=device, dtype=old_conf.dtype)

                    if old_valid.any():
                        old_idx = old_perm[old_pos[old_valid]]
                        old_coords_u[old_valid] = old_coords[old_idx]
                        old_sem_u[old_valid] = old_sem[old_idx]
                        old_conf_u[old_valid] = old_conf[old_idx]

                    if new_valid.any():
                        new_idx = new_perm[new_pos[new_valid]]
                        new_coords_u[new_valid] = current_coords[new_idx]
                        new_sem_u[new_valid] = semantic_probs[new_idx]
                        new_conf_u[new_valid] = confidences[new_idx]

                    chosen_coords = torch.where(new_valid.unsqueeze(1), new_coords_u, old_coords_u)

                    if self.fusion_strategy == 'overwrite':
                        choose_new = new_valid
                        fused_sem = torch.where(choose_new.unsqueeze(1), new_sem_u, old_sem_u)
                        fused_conf = torch.where(choose_new, new_conf_u, old_conf_u)
                        fused_logits = None

                    elif self.fusion_strategy == 'high_conf':
                        old_raw_conf = old_sem_u.max(dim=1).values
                        new_raw_conf = new_sem_u.max(dim=1).values
                        choose_new = new_valid & (~old_valid | (new_raw_conf >= old_raw_conf))
                        fused_sem = torch.where(choose_new.unsqueeze(1), new_sem_u, old_sem_u)
                        fused_conf = torch.where(choose_new, new_conf_u, old_conf_u)
                        fused_logits = None

                    elif self.fusion_strategy == 'weighted_sum_on_logit':
                        if old_logits is None:
                            old_logits = torch.log(torch.clamp(old_sem, min=1e-6))
                        if semantic_logits is None:
                            semantic_logits = torch.log(torch.clamp(semantic_probs, min=1e-6))

                        old_logits_u = torch.zeros((num_voxels, num_classes), device=device, dtype=old_logits.dtype)
                        new_logits_u = torch.zeros((num_voxels, num_classes), device=device, dtype=old_logits.dtype)

                        if old_valid.any():
                            old_idx = old_perm[old_pos[old_valid]]
                            old_logits_u[old_valid] = old_logits[old_idx]
                        if new_valid.any():
                            new_idx = new_perm[new_pos[new_valid]]
                            new_logits_u[new_valid] = semantic_logits[new_idx]

                        fused_logits = torch.zeros_like(old_logits_u)
                        fused_conf = torch.zeros_like(old_conf_u)
                        only_old = old_valid & (~new_valid)
                        only_new = new_valid & (~old_valid)
                        both = old_valid & new_valid

                        fused_logits[only_old] = old_logits_u[only_old]
                        fused_logits[only_new] = new_logits_u[only_new]
                        fused_conf[only_old] = old_conf_u[only_old]
                        fused_conf[only_new] = new_conf_u[only_new]

                        if both.any():
                            denom = torch.clamp(old_conf_u[both] + new_conf_u[both], min=1e-6).unsqueeze(1)
                            fused_logits[both] = (old_logits_u[both] * old_conf_u[both].unsqueeze(1) +
                                                  new_logits_u[both] * new_conf_u[both].unsqueeze(1)) / denom
                            fused_conf[both] = old_conf_u[both] + new_conf_u[both]

                        fused_sem = torch.softmax(fused_logits, dim=1)

                    else:  # weighted_sum_on_prob
                        # Keep legacy behavior for metric alignment:
                        # merge all old/new samples first, then group-by voxel with index_add.
                        coords_all = torch.cat([old_coords, current_coords], dim=0)
                        all_hash = torch.cat([old_hash, current_hash], dim=0)
                        all_sem = torch.cat([old_sem, semantic_probs], dim=0)
                        all_conf = torch.cat([old_conf, confidences], dim=0)

                        unique_hash_legacy, inverse = torch.unique(all_hash, return_inverse=True)
                        num_voxels_legacy = unique_hash_legacy.shape[0]

                        fused_sem = torch.zeros((num_voxels_legacy, num_classes), device=device, dtype=old_sem.dtype)
                        fused_conf = torch.zeros((num_voxels_legacy, ), device=device, dtype=old_conf.dtype)
                        fused_logits = None

                        weighted_sem = all_sem * all_conf.unsqueeze(1)
                        fused_sem.index_add_(0, inverse, weighted_sem)
                        fused_conf.index_add_(0, inverse, all_conf)
                        fused_sem = fused_sem / torch.clamp(fused_conf.unsqueeze(1), min=1e-6)

                        sorted_inverse, perm = torch.sort(inverse)
                        mask = torch.ones_like(sorted_inverse, dtype=torch.bool)
                        mask[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
                        chosen_coords = coords_all[perm[mask]]

                    self.global_coords[batch_idx] = chosen_coords
                    self.global_semantics[batch_idx] = fused_sem
                    self.global_confidences[batch_idx] = fused_conf
                    self.global_logits[batch_idx] = fused_logits

    def get_global_occ(self, scenemetas):
        assert isinstance(scenemetas, list), 'scenemetas must be a list'
        assert len(scenemetas) == len(self.scene_names), 'len(scenemetas) must equal len(self.scene_names)'

        selected_indices = list(range(len(scenemetas)))
        for i, item in enumerate(scenemetas):
            assert isinstance(item, dict), f'scenemetas[{i}] must be a dict'
            assert 'scene_name' in item, f'scenemetas[{i}] must contain scene_name'
            assert item['scene_name'] == self.scene_names[i], (
                f'scenemetas[{i}][scene_name] does not match self.scene_names[{i}]')

        results = []
        for batch_idx in selected_indices:
            if not self.colmap_mode:
                # For evaluation, we directly create the global scene based on the scene size prior.
                coords = self.global_coords[batch_idx]
                semantic_probs = self.global_semantics[batch_idx]
                scene_dim = scenemetas[batch_idx]['global_scene_dim']
                dim_x, dim_y, dim_z = int(scene_dim[0]), int(scene_dim[1]), int(scene_dim[2])

                global_xyz = scenemetas[batch_idx]['global_pts']
                global_origin = self.global_scene_origins[batch_idx]
                voxel_mapping = ((global_xyz - global_origin.unsqueeze(0)) / self.voxel_size).long()

                if len(coords) == 0:
                    num_classes = int(self.head.out_channel) if hasattr(self.head, 'out_channel') else 1
                    device = torch.device('cpu')
                    global_semantic_field = torch.zeros((num_classes, dim_x, dim_y, dim_z), device=device, dtype=torch.float32)
                    observed_mask = torch.zeros((dim_x, dim_y, dim_z), device=device, dtype=torch.bool)
                else:
                    num_classes = semantic_probs.shape[1]
                    device = semantic_probs.device
                    global_semantic_field = torch.zeros((num_classes, dim_x, dim_y, dim_z),
                                                        device=device,
                                                        dtype=semantic_probs.dtype)
                    observed_mask = torch.zeros((dim_x, dim_y, dim_z), device=device, dtype=torch.bool)

                    coords = coords.to(device=device, dtype=torch.long)
                    semantic_probs = semantic_probs.to(device=device)

                    valid_mask = ((coords[:, 0] >= 0) & (coords[:, 0] < dim_x) & (coords[:, 1] >= 0) & (coords[:, 1] < dim_y) &
                                  (coords[:, 2] >= 0) & (coords[:, 2] < dim_z))
                    coords = coords[valid_mask]
                    semantic_probs = semantic_probs[valid_mask]

                    if coords.shape[0] > 0:
                        gx, gy, gz = coords[:, 0], coords[:, 1], coords[:, 2]
                        global_semantic_field[:, gx, gy, gz] = semantic_probs.permute(1, 0)
                        observed_mask[gx, gy, gz] = True

                predict = torch.argmax(global_semantic_field, dim=0).long()

                if hasattr(self.head, 'empty_idx'):
                    empty_idx = int(self.head.empty_idx)
                    predict[~observed_mask] = empty_idx

                # Reorder predict according to voxel_mapping to match global_labels storage order
                vx = voxel_mapping[..., 0]
                vy = voxel_mapping[..., 1]
                vz = voxel_mapping[..., 2]

                valid = (vx >= 0) & (vx < dim_x) & (vy >= 0) & (vy < dim_y) & (vz >= 0) & (vz < dim_z)
                predict_reordered = torch.zeros_like(voxel_mapping[..., 0], dtype=predict.dtype, device=predict.device)

                if hasattr(self.head, 'empty_idx'):
                    predict_reordered[~valid] = int(self.head.empty_idx)
                predict_reordered[valid] = predict[vx[valid], vy[valid], vz[valid]]
                predict = predict_reordered

                label = scenemetas[batch_idx]['global_labels'].clone()
                label[label == 0] = 12

                scene_result_dict = {
                    'scene_name': self.scene_names[batch_idx],
                    'predict': predict,
                    'label': label,
                    'mask': self.global_masks_thistime[batch_idx],
                    'global_pts': scenemetas[batch_idx]['global_pts'],
                    'colmap_mode': False,
                }
                results.append(scene_result_dict)
            else:
                # Dynamically build the global scene based on the updated coordinates and semantics. This is more like the real online setting.
                coords = self.global_coords[batch_idx]
                semantic_probs = self.global_semantics[batch_idx]

                if coords is None or len(coords) == 0:
                    predict = torch.zeros((1, 1, 1), dtype=torch.long)
                    label = torch.ones((1, 1, 1), dtype=torch.long) * 12
                    mask = torch.zeros((1, 1, 1), dtype=torch.bool)
                    global_pts = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
                else:
                    min_coords = coords.min(dim=0).values
                    max_coords = coords.max(dim=0).values
                    dim_x, dim_y, dim_z = (max_coords - min_coords + 1).tolist()

                    num_classes = semantic_probs.shape[1]
                    device = semantic_probs.device

                    global_semantic_field = torch.zeros((num_classes, dim_x, dim_y, dim_z),
                                                        device=device,
                                                        dtype=semantic_probs.dtype)
                    observed_mask = torch.ones((dim_x, dim_y, dim_z), device=device, dtype=torch.bool)

                    gx = coords[:, 0] - min_coords[0]
                    gy = coords[:, 1] - min_coords[1]
                    gz = coords[:, 2] - min_coords[2]

                    global_semantic_field[:, gx, gy, gz] = semantic_probs.permute(1, 0)
                    # observed_mask[gx, gy, gz] = True

                    predict = torch.argmax(global_semantic_field, dim=0).long()
                    if hasattr(self.head, 'empty_idx'):
                        empty_idx = int(self.head.empty_idx)
                        predict[~observed_mask] = empty_idx

                    label = torch.ones_like(predict) * 12
                    mask = observed_mask.cpu()

                    global_origin = self.global_scene_origins[batch_idx]
                    global_origin_tensor = global_origin.clone().detach() if torch.is_tensor(global_origin) else torch.tensor(
                        global_origin, device=device)
                    global_origin_tensor = global_origin_tensor.to(device)

                    x_grid = torch.arange(dim_x, device=device) + min_coords[0]
                    y_grid = torch.arange(dim_y, device=device) + min_coords[1]
                    z_grid = torch.arange(dim_z, device=device) + min_coords[2]
                    xx, yy, zz = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

                    global_pts = torch.stack([xx, yy, zz], dim=-1).float() * self.voxel_size + global_origin_tensor

                scene_result_dict = {
                    'scene_name': self.scene_names[batch_idx],
                    'predict': predict,
                    'label': label,
                    'mask': mask,
                    'global_pts': global_pts,
                    'colmap_mode': True,
                }
                results.append(scene_result_dict)

        return results

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        scene_metas=None,
        **kwargs,
    ):
        B, F, N, C, H, W = imgs.shape
        assert grad_frames is None
        assert F == 1, 'Only F=1 supported for now'

        output_dict = super().forward(
            imgs=imgs,
            metas=metas,
            points=points,
            label=label,
            grad_frames=grad_frames,
            test_mode=test_mode,
        )

        if scene_metas is not None:
            output_dict['scene_update_payload'] = self._build_scene_update_payload(output_dict,
                                                                                   metas,
                                                                                   scene_metas,
                                                                                   image_hw=(H, W))
        else:
            output_dict['scene_update_payload'] = None

        return output_dict

    def loss(self, output_dict, metas, label):
        total_loss = 0.
        loss_dict = dict()

        local_loss, local_loss_dict = super().loss(output_dict, metas, label)
        total_loss += local_loss
        loss_dict.update(local_loss_dict)

        scene_update_payload = output_dict.get('scene_update_payload', None)
        if self.local_fuser is not None:
            fusion_loss, fusion_loss_dict = self.local_fuser.loss(scene_update_payload, self.global_labels, self.global_xyzs)
        else:
            fusion_loss = self._build_zero_fuser_loss(output_dict['ce_input'])
            fusion_loss_dict = {
                'fusion_loss': fusion_loss.detach().item(),
                'fusion_valid_points': 0,
                'fusion_batches': 0,
            }
        total_loss += fusion_loss
        loss_dict.update(fusion_loss_dict)

        return total_loss, loss_dict
