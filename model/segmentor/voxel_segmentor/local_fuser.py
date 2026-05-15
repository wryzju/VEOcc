import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


@MODELS.register_module()
class LocalLogitFusion(nn.Module):

    def __init__(
        self,
        in_channels=96,
        num_classes=13,
        pe_num_freqs=4,
        hidden_channels=256,
        min_valid_points=1,
        ignore_label=0,
        class_frequencies=None,
        loss_weight=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pe_num_freqs = pe_num_freqs
        self.min_valid_points = min_valid_points
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.class_frequencies = class_frequencies

        pe_dim_per_xyz_or_uvd = 3 * (2 * pe_num_freqs)
        self.pos_proj = nn.Sequential(
            nn.Linear(pe_dim_per_xyz_or_uvd * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )

        self.conf_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

        coord_dim = 3 * (2 * pe_num_freqs)
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        self.logit_proj = nn.Sequential(
            nn.Linear(num_classes, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        self.conf_mlp = nn.Sequential(
            nn.Linear(in_channels * 12, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),
        )
        self.last_fused_points_count = 0

        if class_frequencies is not None:
            class_freq = torch.as_tensor(class_frequencies, dtype=torch.float32)
            class_weights = 1.0 / torch.log(class_freq + 0.001)
        else:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)

    def _to_tensor(self, data, device, dtype=None):
        if torch.is_tensor(data):
            if dtype is None:
                return data.to(device=device)
            return data.to(device=device, dtype=dtype)
        if dtype is None:
            return torch.tensor(data, device=device)
        return torch.tensor(data, device=device, dtype=dtype)

    def _positional_encoding(self, x):
        if x.numel() == 0:
            return x.new_zeros((0, x.shape[-1] * 2 * self.pe_num_freqs))
        embeds = []
        for i in range(self.pe_num_freqs):
            freq = 2.0**i
            embeds.append(torch.sin(freq * x))
            embeds.append(torch.cos(freq * x))
        return torch.cat(embeds, dim=-1)

    def _world_to_camera_uvd(self, world_xyz, meta, image_hw):
        device = world_xyz.device
        dtype = world_xyz.dtype
        cam2world = self._to_tensor(meta['cam2world'], device=device, dtype=dtype)
        cam_k = self._to_tensor(meta['cam_k'], device=device, dtype=dtype)
        if cam_k.shape[0] == 4 and cam_k.shape[1] == 4:
            cam_k = cam_k[:3, :3]

        cam_pos = cam2world[:3, 3]
        rot = cam2world[:3, :3]
        points_cam = (world_xyz - cam_pos.unsqueeze(0)) @ rot

        z = points_cam[:, 2].clamp(min=1e-4)
        fx, fy = cam_k[0, 0], cam_k[1, 1]
        cx, cy = cam_k[0, 2], cam_k[1, 2]
        u = fx * (points_cam[:, 0] / z) + cx
        v = fy * (points_cam[:, 1] / z) + cy

        h, w = image_hw
        u_norm = u / max(float(w - 1), 1.0)
        v_norm = v / max(float(h - 1), 1.0)
        d_norm = z / torch.clamp(z.max(), min=1.0)
        uvd = torch.stack([u_norm, v_norm, d_norm], dim=-1)
        return points_cam, uvd

    def _extract_image_hw(self, meta):
        img_shape = meta.get('img_shape', None)
        if img_shape is None:
            return 480.0, 640.0
        if torch.is_tensor(img_shape):
            values = img_shape.flatten().tolist()
        elif isinstance(img_shape, (list, tuple)):
            if len(img_shape) > 0 and isinstance(img_shape[0], (list, tuple)):
                values = list(img_shape[0])
            else:
                values = list(img_shape)
        else:
            return 480.0, 640.0
        if len(values) >= 2:
            return float(values[0]), float(values[1])
        return 480.0, 640.0

    def _sample_prev_by_world(self, prev_tensor_bcdhw, world_xyz, prev_meta):
        device = world_xyz.device
        dtype = world_xyz.dtype
        prev_origin = self._to_tensor(prev_meta['vox_origin'], device=device, dtype=dtype)
        prev_scene_size = self._to_tensor(prev_meta['scene_size'], device=device, dtype=dtype)

        size_x, size_y, size_z = prev_tensor_bcdhw.shape[-3:]
        x_idx = (world_xyz[:, 0] - prev_origin[0]) / prev_scene_size[0] * (size_x - 1)
        y_idx = (world_xyz[:, 1] - prev_origin[1]) / prev_scene_size[1] * (size_y - 1)
        z_idx = (world_xyz[:, 2] - prev_origin[2]) / prev_scene_size[2] * (size_z - 1)

        valid = (x_idx >= 0) & (x_idx <= (size_x - 1)) & (y_idx >= 0) & (y_idx <= (size_y - 1)) & (z_idx >= 0) & (z_idx
                                                                                                                  <= (size_z - 1))
        if valid.sum() == 0:
            return valid, None, None

        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        z_idx = z_idx[valid]
        x_norm = 2.0 * x_idx / max(size_x - 1, 1) - 1.0
        y_norm = 2.0 * y_idx / max(size_y - 1, 1) - 1.0
        z_norm = 2.0 * z_idx / max(size_z - 1, 1) - 1.0
        sample_grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).view(1, -1, 1, 1, 3)

        tensor_ncdhw = prev_tensor_bcdhw.permute(0, 1, 4, 3, 2).contiguous()
        sampled = F.grid_sample(
            tensor_ncdhw,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ).squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)
        return valid, sampled, sample_grid

    def _zero_backward_loss(self, reference_tensor):
        zero_loss = reference_tensor.sum() * 0.0
        grad_anchor = None
        for param in self.parameters():
            if param.requires_grad:
                if grad_anchor is None:
                    grad_anchor = param.sum() * 0.0
                else:
                    grad_anchor = grad_anchor + param.sum() * 0.0
        if grad_anchor is not None:
            zero_loss = zero_loss + grad_anchor
        return zero_loss

    def _hash_world_coords(self, coords):
        if coords is None or coords.numel() == 0:
            return None
        coords = coords.to(dtype=torch.float64)
        quant = torch.round(coords * 10000.0).to(torch.int64)
        return quant[:, 0] * 73856093 + quant[:, 1] * 19349663 + quant[:, 2] * 83492791

    def _lookup_gt_labels(self, fused_coords, global_xyz, global_label):
        if fused_coords is None or fused_coords.numel() == 0:
            return None

        device = fused_coords.device
        fused_coords = fused_coords.to(device=device)
        global_xyz = self._to_tensor(global_xyz, device=device, dtype=fused_coords.dtype)
        global_label = self._to_tensor(global_label, device=device, dtype=torch.long)

        if global_xyz.dim() == 4 and global_xyz.shape[-1] == 3:
            global_xyz = global_xyz.reshape(-1, 3)
        elif global_xyz.dim() == 2 and global_xyz.shape[-1] == 3:
            pass
        else:
            return None

        label_flat = global_label.reshape(-1)
        if global_xyz.shape[0] != label_flat.shape[0]:
            return None

        global_hash = self._hash_world_coords(global_xyz)
        fused_hash = self._hash_world_coords(fused_coords)
        if global_hash is None or fused_hash is None:
            return None

        sorted_hash, sorted_idx = torch.sort(global_hash)
        search_pos = torch.searchsorted(sorted_hash, fused_hash)
        search_pos = torch.clamp(search_pos, max=sorted_hash.shape[0] - 1)

        matched = sorted_hash[search_pos] == fused_hash
        gt_labels = torch.full((fused_coords.shape[0], ), self.ignore_label, device=device, dtype=torch.long)
        if matched.any():
            gt_labels[matched] = label_flat[sorted_idx[search_pos[matched]]]
        return gt_labels

    def _extract_fused_labels_from_scene(self, fused_coords, global_xyz, global_labels):
        gt_labels = self._lookup_gt_labels(fused_coords, global_xyz, global_labels)
        if gt_labels is None:
            return None
        return gt_labels

    def fuse(self, curr_output, curr_metas, prev_output, prev_metas, valid_pair_mask=None):
        curr_logits = curr_output['ce_input']
        curr_feat = curr_output['img_voxel_feature']
        curr_labels = curr_output['ce_label']
        curr_fov = curr_output['fov_mask']

        prev_logits = prev_output['ce_input']
        prev_feat = prev_output['img_voxel_feature']
        prev_fov = prev_output['fov_mask']

        device = curr_logits.device
        batch_size = curr_logits.shape[0]
        if valid_pair_mask is None:
            valid_pair_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        else:
            valid_pair_mask = valid_pair_mask.to(device=device, dtype=torch.bool)

        fused_logits = curr_logits.clone()
        total_loss = self._zero_backward_loss(curr_logits)
        valid_batches = 0
        fused_points_total = 0

        for b in range(batch_size):
            if not valid_pair_mask[b]:
                continue

            curr_scene = str(curr_metas[b].get('name', '')).split('/')[0]
            prev_scene = str(prev_metas[b].get('name', '')).split('/')[0]
            if curr_scene != prev_scene:
                continue

            curr_fov_mask = curr_fov[b].to(device=device, dtype=torch.bool)
            if curr_fov_mask.sum() < self.min_valid_points:
                continue

            occ_xyz_curr = self._to_tensor(curr_metas[b]['occ_xyz'], device=device, dtype=curr_feat.dtype)
            world_xyz_all = occ_xyz_curr[curr_fov_mask]
            if world_xyz_all.shape[0] < self.min_valid_points:
                continue

            curr_idx = torch.nonzero(curr_fov_mask, as_tuple=False)
            curr_feat_pts = curr_feat[b, :, curr_idx[:, 0], curr_idx[:, 1], curr_idx[:, 2]].transpose(0, 1)
            curr_logit_pts = curr_logits[b, :, curr_idx[:, 0], curr_idx[:, 1], curr_idx[:, 2]].transpose(0, 1)
            curr_label_pts = curr_labels[b, curr_idx[:, 0], curr_idx[:, 1], curr_idx[:, 2]].long()

            valid_world, prev_feat_pts, sample_grid = self._sample_prev_by_world(prev_feat[b:b + 1], world_xyz_all, prev_metas[b])
            if prev_feat_pts is None:
                continue

            world_xyz = world_xyz_all[valid_world]
            curr_idx = curr_idx[valid_world]
            curr_feat_pts = curr_feat_pts[valid_world]
            curr_logit_pts = curr_logit_pts[valid_world]
            curr_label_pts = curr_label_pts[valid_world]

            valid_world_2, prev_logit_pts, sample_grid_logits = self._sample_prev_by_world(prev_logits[b:b + 1], world_xyz,
                                                                                           prev_metas[b])
            if prev_logit_pts is None:
                continue

            world_xyz = world_xyz[valid_world_2]
            curr_idx = curr_idx[valid_world_2]
            curr_feat_pts = curr_feat_pts[valid_world_2]
            curr_logit_pts = curr_logit_pts[valid_world_2]
            curr_label_pts = curr_label_pts[valid_world_2]
            prev_feat_pts = prev_feat_pts[valid_world_2]
            prev_logit_pts = prev_logit_pts[valid_world_2]

            prev_fov_ncdhw = prev_fov[b:b + 1].to(device=device, dtype=curr_feat.dtype).unsqueeze(1).permute(0, 1, 4, 3,
                                                                                                             2).contiguous()
            prev_fov_value = F.grid_sample(
                prev_fov_ncdhw,
                sample_grid_logits,
                mode='nearest',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(0).squeeze(0).squeeze(-1).squeeze(-1)
            prev_fov_valid = prev_fov_value > 0.5
            if prev_fov_valid.sum() < self.min_valid_points:
                continue

            world_xyz = world_xyz[prev_fov_valid]
            curr_idx = curr_idx[prev_fov_valid]
            curr_feat_pts = curr_feat_pts[prev_fov_valid]
            curr_logit_pts = curr_logit_pts[prev_fov_valid]
            curr_label_pts = curr_label_pts[prev_fov_valid]
            prev_feat_pts = prev_feat_pts[prev_fov_valid]
            prev_logit_pts = prev_logit_pts[prev_fov_valid]

            h_curr, w_curr = self._extract_image_hw(curr_metas[b])
            h_prev, w_prev = self._extract_image_hw(prev_metas[b])

            curr_cam_xyz, curr_uvd = self._world_to_camera_uvd(world_xyz, curr_metas[b], image_hw=(h_curr, w_curr))
            prev_cam_xyz, prev_uvd = self._world_to_camera_uvd(world_xyz, prev_metas[b], image_hw=(h_prev, w_prev))

            curr_pe = torch.cat([self._positional_encoding(curr_cam_xyz), self._positional_encoding(curr_uvd)], dim=-1)
            prev_pe = torch.cat([self._positional_encoding(prev_cam_xyz), self._positional_encoding(prev_uvd)], dim=-1)
            curr_enh = curr_feat_pts + self.pos_proj(curr_pe)
            prev_enh = prev_feat_pts + self.pos_proj(prev_pe)

            feat_curr_input = torch.cat([curr_enh, curr_enh - prev_enh], dim=-1)
            feat_prev_input = torch.cat([prev_enh, prev_enh - curr_enh], dim=-1)
            logit_curr = self.conf_mlp(feat_curr_input)
            logit_prev = self.conf_mlp(feat_prev_input)
            conf_logits = torch.cat([logit_curr, logit_prev], dim=-1)
            conf_prob = torch.softmax(conf_logits, dim=-1)

            w_curr_pts = conf_prob[:, 0:1]
            w_prev_pts = conf_prob[:, 1:2]
            fused_pts = w_curr_pts * curr_logit_pts + w_prev_pts * prev_logit_pts

            fused_logits[b, :, curr_idx[:, 0], curr_idx[:, 1], curr_idx[:, 2]] = fused_pts.transpose(0, 1)

            valid_label = curr_label_pts != self.ignore_label
            if valid_label.sum() > 0:
                loss_pts = F.cross_entropy(fused_pts[valid_label], curr_label_pts[valid_label], reduction='mean')
                total_loss = total_loss + self.loss_weight * loss_pts
                valid_batches += 1

            fused_points_total += int(curr_idx.shape[0])

        if valid_batches > 0:
            total_loss = total_loss / valid_batches
        else:
            total_loss = self._zero_backward_loss(curr_logits)

        return {
            'fused_ce_input': fused_logits,
            'fuser_loss': total_loss,
            'fused_points_count': fused_points_total,
        }

    def _parse_fuser_input(self, fuser_input):
        if not isinstance(fuser_input, (list, tuple)) or len(fuser_input) != 4:
            raise TypeError('fuser_input must be a list/tuple of [sampled_logits, sampled_feature, coords_to_update, metas]')

        sampled_logits, sampled_feature, coords_to_update, metas = fuser_input
        if metas is None:
            metas = {}
        return sampled_logits, sampled_feature, coords_to_update, metas

    def _as_point_major(self, tensor, expected_dim=None):
        if tensor is None:
            return None, False
        if tensor.dim() != 2:
            raise ValueError('Expected a 2D tensor for point-wise fusion inputs.')

        transposed = False
        if expected_dim is not None:
            if tensor.shape[0] == expected_dim:
                tensor = tensor.transpose(0, 1).contiguous()
                transposed = True
            elif tensor.shape[1] != expected_dim:
                raise ValueError(f'Input tensor shape {tuple(tensor.shape)} is incompatible with expected dim {expected_dim}.')
        return tensor, transposed

    def _normalize_coords(self, coords_to_update, metas):
        if coords_to_update is None:
            return None
        if coords_to_update.numel() == 0:
            return coords_to_update.new_zeros((0, 3))

        device = coords_to_update.device
        dtype = torch.float32

        origin = metas.get('vox_origin', None) if isinstance(metas, dict) else None
        scene_size = metas.get('scene_size', None) if isinstance(metas, dict) else None

        if origin is None or scene_size is None:
            return coords_to_update.to(device=device, dtype=dtype)

        origin = self._to_tensor(origin, device=device, dtype=dtype)
        scene_size = self._to_tensor(scene_size, device=device, dtype=dtype)
        return ((coords_to_update.to(device=device, dtype=dtype) - origin.unsqueeze(0)) /
                torch.clamp(scene_size.unsqueeze(0), min=1e-6))

    def _build_pair_repr(self, sampled_logits_this, sampled_feature_this, coords_this, metas_this, sampled_logits_prev,
                         sampled_feature_prev, coords_prev, metas_prev):
        coords_this = self._normalize_coords(coords_this, metas_this)
        coords_prev = self._normalize_coords(coords_prev, metas_prev)

        coords_this = coords_this.to(dtype=self.coord_proj[0].weight.dtype)
        coords_prev = coords_prev.to(dtype=self.coord_proj[0].weight.dtype)

        coords_embed_this = self.coord_proj(self._positional_encoding(coords_this))
        coords_embed_prev = self.coord_proj(self._positional_encoding(coords_prev))

        logit_feat_this = self.logit_proj(sampled_logits_this)
        logit_feat_prev = self.logit_proj(sampled_logits_prev)
        feat_feat_this = self.feat_proj(sampled_feature_this)
        feat_feat_prev = self.feat_proj(sampled_feature_prev)

        curr_repr = torch.cat([logit_feat_this, feat_feat_this, coords_embed_this], dim=-1)
        prev_repr = torch.cat([logit_feat_prev, feat_feat_prev, coords_embed_prev], dim=-1)
        pair_repr = torch.cat([curr_repr, prev_repr, curr_repr - prev_repr, curr_repr * prev_repr], dim=-1)
        return curr_repr, prev_repr, pair_repr

    def fuse(self, fuser_input_this, fuser_input_prev):
        sampled_logits_this, sampled_feature_this, coords_this, metas_this = self._parse_fuser_input(fuser_input_this)
        sampled_logits_prev, sampled_feature_prev, coords_prev, metas_prev = self._parse_fuser_input(fuser_input_prev)

        sampled_logits_this, this_transposed = self._as_point_major(sampled_logits_this, expected_dim=self.num_classes)
        sampled_logits_prev, _ = self._as_point_major(sampled_logits_prev, expected_dim=self.num_classes)
        sampled_feature_this, _ = self._as_point_major(sampled_feature_this, expected_dim=self.in_channels)
        sampled_feature_prev, _ = self._as_point_major(sampled_feature_prev, expected_dim=self.in_channels)

        if sampled_logits_this is None or sampled_logits_prev is None:
            if sampled_logits_this is None:
                return sampled_logits_prev.transpose(0, 1).contiguous() if sampled_logits_prev is not None else None
            return sampled_logits_this.transpose(0, 1).contiguous() if this_transposed else sampled_logits_this

        num_points = min(sampled_logits_this.shape[0], sampled_logits_prev.shape[0], sampled_feature_this.shape[0],
                         sampled_feature_prev.shape[0], coords_this.shape[0], coords_prev.shape[0])
        if num_points == 0:
            return sampled_logits_this.transpose(0, 1).contiguous() if this_transposed else sampled_logits_this

        sampled_logits_this = sampled_logits_this[:num_points]
        sampled_logits_prev = sampled_logits_prev[:num_points]
        sampled_feature_this = sampled_feature_this[:num_points]
        sampled_feature_prev = sampled_feature_prev[:num_points]
        coords_this = coords_this[:num_points]
        coords_prev = coords_prev[:num_points]

        _, _, pair_repr = self._build_pair_repr(
            sampled_logits_this,
            sampled_feature_this,
            coords_this,
            metas_this,
            sampled_logits_prev,
            sampled_feature_prev,
            coords_prev,
            metas_prev,
        )

        weights = torch.softmax(self.conf_mlp(pair_repr), dim=-1)
        fused_logits = weights[:, 0:1] * sampled_logits_this + weights[:, 1:2] * sampled_logits_prev
        self.last_fused_points_count = int(num_points)

        if this_transposed:
            return fused_logits.transpose(0, 1).contiguous()
        return fused_logits

    def loss(self, scene_update_payload, global_labels, global_xyzs):
        if scene_update_payload is None:
            reference = next(self.parameters())
            zero_loss = self._zero_backward_loss(reference)
            return zero_loss, {
                'fusion_loss': zero_loss.detach().item(),
                'fusion_valid_points': 0,
                'fusion_batches': 0,
            }

        reference = next(self.parameters())
        total_loss = self._zero_backward_loss(reference)
        valid_batches = 0
        valid_points = 0

        batch_size = min(len(scene_update_payload), len(global_labels), len(global_xyzs))
        for batch_idx in range(batch_size):
            payload = scene_update_payload[batch_idx]
            fused_coords = payload.get('fused_coords', None)
            fused_logits = payload.get('fused_logits', None)
            if fused_coords is None or fused_logits is None:
                continue
            if fused_coords.numel() == 0 or fused_logits.numel() == 0:
                continue

            fused_logits, _ = self._as_point_major(fused_logits, expected_dim=self.num_classes)
            gt_labels = self._extract_fused_labels_from_scene(fused_coords, global_xyzs[batch_idx], global_labels[batch_idx])
            if gt_labels is None:
                continue

            valid_mask = gt_labels != self.ignore_label
            if valid_mask.sum() == 0:
                continue

            loss_b = F.cross_entropy(
                fused_logits[valid_mask],
                gt_labels[valid_mask],
                weight=self.class_weights.to(device=fused_logits.device, dtype=fused_logits.dtype),
                ignore_index=self.ignore_label,
                reduction='mean',
            )
            total_loss = total_loss + self.loss_weight * loss_b
            valid_batches += 1
            valid_points += int(valid_mask.sum().item())

        if valid_batches > 0:
            total_loss = total_loss / valid_batches

        return total_loss, {
            'fusion_loss': total_loss.detach().item(),
            'fusion_valid_points': valid_points,
            'fusion_batches': valid_batches,
        }
