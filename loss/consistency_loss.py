import torch
import torch.nn.functional as F


def _to_tensor(data, device, dtype=None):
    if torch.is_tensor(data):
        if dtype is None:
            return data.to(device=device)
        return data.to(device=device, dtype=dtype)
    if dtype is None:
        return torch.tensor(data, device=device)
    return torch.tensor(data, device=device, dtype=dtype)


def _get_scene_name(meta):
    if 'scene_name' in meta and meta['scene_name'] is not None:
        return str(meta['scene_name'])
    if 'name' in meta and meta['name'] is not None:
        name = str(meta['name'])
        if '/' in name:
            return name.split('/')[0]
        return name
    return None


def _get_image_hw(meta):
    img_shape = meta.get('img_shape', None)
    if img_shape is None:
        return 480.0, 640.0

    if torch.is_tensor(img_shape):
        values = img_shape.flatten().tolist()
    else:
        values = list(img_shape)

    if len(values) >= 2:
        return float(values[0]), float(values[1])
    return 480.0, 640.0


def _compute_uvd_confidence(sample_xyz_world, meta, image_hw, min_conf=0.01, depth_decay=0.10, uv_decay=1.50):
    if sample_xyz_world is None:
        return None
    if sample_xyz_world.numel() == 0:
        return sample_xyz_world.new_zeros((0, ))

    device = sample_xyz_world.device
    dtype = sample_xyz_world.dtype
    height, width = image_hw

    cam2world = _to_tensor(meta['cam2world'], device=device, dtype=dtype)
    cam_k = _to_tensor(meta['cam_k'], device=device, dtype=dtype)
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

    depth_decay_t = F.softplus(torch.tensor(depth_decay, device=device, dtype=dtype))
    uv_decay_t = F.softplus(torch.tensor(uv_decay, device=device, dtype=dtype))

    depth_conf = torch.exp(-depth_decay_t * z)

    dist_left = u
    dist_right = (width - 1.0) - u
    dist_top = v
    dist_bottom = (height - 1.0) - v
    dist_to_border = torch.min(torch.min(dist_left, dist_right), torch.min(dist_top, dist_bottom))
    max_dist = max(min(width, height) * 0.5, 1.0)
    dist_to_border_norm = torch.clamp(dist_to_border / max_dist, min=0.0, max=1.0)
    border_proximity = 1.0 - dist_to_border_norm
    uv_conf = torch.exp(-uv_decay_t * border_proximity)

    in_front_mask = z_raw > 1e-4
    in_frame_mask = (u >= 0) & (u <= (width - 1.0)) & (v >= 0) & (v <= (height - 1.0))

    final_conf = depth_conf * uv_conf
    min_conf_t = torch.tensor(min_conf, device=device, dtype=dtype)
    final_conf = torch.clamp(final_conf, min=min_conf_t, max=1.0)
    final_conf = torch.where(in_front_mask & in_frame_mask, final_conf, min_conf_t)
    return final_conf


def compute_consistency_loss_from_pair(
    curr_output_dict,
    curr_metas,
    prev_output_dict,
    prev_metas,
    valid_pair_mask=None,
):
    device = curr_output_dict['img_voxel_feature'].device
    zero = curr_output_dict['img_voxel_feature'].new_tensor(0.0)

    curr_feat = curr_output_dict['img_voxel_feature']
    prev_feat = prev_output_dict['img_voxel_feature']
    curr_fov = curr_output_dict['fov_mask']
    prev_fov = prev_output_dict['fov_mask']

    batch_size = min(curr_feat.shape[0], prev_feat.shape[0], len(curr_metas), len(prev_metas))
    if batch_size == 0:
        return zero

    if valid_pair_mask is None:
        valid_pair_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
    else:
        valid_pair_mask = valid_pair_mask.to(device=device, dtype=torch.bool)

    loss_sum = zero
    weight_sum = zero

    for batch_idx in range(batch_size):
        if not valid_pair_mask[batch_idx]:
            continue

        meta_curr = curr_metas[batch_idx]
        meta_prev = prev_metas[batch_idx]

        scene_curr = _get_scene_name(meta_curr)
        scene_prev = _get_scene_name(meta_prev)
        if scene_curr is not None and scene_prev is not None and scene_curr != scene_prev:
            continue

        curr_fov_mask = curr_fov[batch_idx].to(device=device, dtype=torch.bool)
        if curr_fov_mask.sum() == 0:
            continue

        occ_xyz_curr = _to_tensor(meta_curr['occ_xyz'], device=device, dtype=curr_feat.dtype)
        world_xyz = occ_xyz_curr[curr_fov_mask]
        if world_xyz.shape[0] == 0:
            continue

        curr_mask_idx = torch.nonzero(curr_fov_mask, as_tuple=False)
        curr_point_feat = curr_feat[batch_idx, :, curr_mask_idx[:, 0], curr_mask_idx[:, 1], curr_mask_idx[:, 2]].transpose(0, 1)

        prev_origin = _to_tensor(meta_prev['vox_origin'], device=device, dtype=curr_feat.dtype)
        prev_scene_size = _to_tensor(meta_prev['scene_size'], device=device, dtype=curr_feat.dtype)

        size_x, size_y, size_z = prev_feat.shape[-3:]
        x_idx = (world_xyz[:, 0] - prev_origin[0]) / prev_scene_size[0] * (size_x - 1)
        y_idx = (world_xyz[:, 1] - prev_origin[1]) / prev_scene_size[1] * (size_y - 1)
        z_idx = (world_xyz[:, 2] - prev_origin[2]) / prev_scene_size[2] * (size_z - 1)

        valid = (x_idx >= 0) & (x_idx <= (size_x - 1)) & (y_idx >= 0) & (y_idx <= (size_y - 1)) & (z_idx >= 0) & (z_idx
                                                                                                                  <= (size_z - 1))
        if valid.sum() == 0:
            continue

        world_xyz = world_xyz[valid]
        curr_point_feat = curr_point_feat[valid]
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        z_idx = z_idx[valid]

        x_norm = 2.0 * x_idx / max(size_x - 1, 1) - 1.0
        y_norm = 2.0 * y_idx / max(size_y - 1, 1) - 1.0
        z_norm = 2.0 * z_idx / max(size_z - 1, 1) - 1.0
        sample_grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).view(1, -1, 1, 1, 3)

        prev_feat_ncdhw = prev_feat[batch_idx:batch_idx + 1].permute(0, 1, 4, 3, 2).contiguous()
        sampled_prev_feat = F.grid_sample(
            prev_feat_ncdhw,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ).squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)

        prev_fov_ncdhw = prev_fov[batch_idx:batch_idx + 1].to(device=device,
                                                              dtype=curr_feat.dtype).unsqueeze(1).permute(0, 1, 4, 3,
                                                                                                          2).contiguous()
        prev_fov_value = F.grid_sample(
            prev_fov_ncdhw,
            sample_grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True,
        ).squeeze(0).squeeze(0).squeeze(-1).squeeze(-1)
        prev_fov_valid = prev_fov_value > 0.5
        if prev_fov_valid.sum() == 0:
            continue

        world_xyz = world_xyz[prev_fov_valid]
        curr_point_feat = curr_point_feat[prev_fov_valid]
        sampled_prev_feat = sampled_prev_feat[prev_fov_valid]

        cos_sim = F.cosine_similarity(curr_point_feat, sampled_prev_feat, dim=1, eps=1e-6)
        point_loss = 1.0 - cos_sim

        # image_hw_curr = _get_image_hw(meta_curr)
        # image_hw_prev = _get_image_hw(meta_prev)
        # conf_curr = _compute_uvd_confidence(world_xyz, meta_curr, image_hw_curr)
        # conf_prev = _compute_uvd_confidence(world_xyz, meta_prev, image_hw_prev)
        # conf_weight = conf_curr * conf_prev

        image_hw_prev = _get_image_hw(meta_prev)
        conf_prev = _compute_uvd_confidence(world_xyz, meta_prev, image_hw_prev)
        conf_weight = conf_prev

        loss_sum = loss_sum + (point_loss * conf_weight).sum()
        weight_sum = weight_sum + conf_weight.sum()

    if weight_sum <= 0:
        return zero

    return loss_sum / (weight_sum + 1e-6)


def compute_consistency_loss(output_dict, metas, loss_cache):
    device = output_dict['img_voxel_feature'].device
    zero = output_dict['img_voxel_feature'].new_tensor(0.0)
    if loss_cache is None:
        return zero
    if 'img_voxel_feature' not in loss_cache or 'fov_mask' not in loss_cache or 'metas' not in loss_cache:
        return zero

    prev_output = {
        'img_voxel_feature': loss_cache['img_voxel_feature'],
        'fov_mask': loss_cache['fov_mask'],
    }
    return compute_consistency_loss_from_pair(
        curr_output_dict=output_dict,
        curr_metas=metas,
        prev_output_dict=prev_output,
        prev_metas=loss_cache['metas'],
    )
