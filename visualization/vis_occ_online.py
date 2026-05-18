#!/usr/bin/env python3
"""Online voxel visualization tool - fixed top-down view for per-scene prediction frames."""

import os
import sys

os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from mayavi import mlab
from PIL import Image


def remove_white_background(image_path):
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    white_areas = (data[:, :, 0] > 240) & (data[:, :, 1]
                                           > 240) & (data[:, :, 2] > 240)
    data[white_areas] = [255, 255, 255, 0]

    new_img = Image.fromarray(data, 'RGBA')
    new_img.save(image_path.replace('.png', '_t.png'))


def load_camera_config(config_path="camera_config_online.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[INFO] Loaded camera configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(
            f"[WARNING] Config file {config_path} not found, using default settings"
        )
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {config_path}, using default settings")
        return None


def estimate_scene_size_from_pcd(pcd_path, voxel_size=0.08):
    """Estimate a stable scene scale from a reference point cloud."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"Reference point cloud is empty: {pcd_path}")

    # voxel_coords = np.floor(points / voxel_size).astype(int)
    voxel_coords = np.floor((points / voxel_size) + 1e-4).astype(int)
    voxel_centers = np.unique(
        voxel_coords, axis=0).astype(float) * voxel_size + voxel_size / 2
    if voxel_centers.size == 0:
        raise ValueError(f"Reference voxel set is empty: {pcd_path}")

    min_bounds = np.min(voxel_centers, axis=0)
    max_bounds = np.max(voxel_centers, axis=0)
    return float(np.max(max_bounds - min_bounds))


def estimate_scene_center_and_size_from_pcd(pcd_path, voxel_size=0.08):
    """Estimate a stable scene center and scale from a reference point cloud."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"Reference point cloud is empty: {pcd_path}")

    voxel_coords = np.floor((points / voxel_size) + 1e-4).astype(int)
    voxel_centers = np.unique(
        voxel_coords, axis=0).astype(float) * voxel_size + voxel_size / 2
    if voxel_centers.size == 0:
        raise ValueError(f"Reference voxel set is empty: {pcd_path}")

    min_bounds = np.min(voxel_centers, axis=0)
    max_bounds = np.max(voxel_centers, axis=0)
    center = (min_bounds + max_bounds) / 2
    scene_size = float(np.max(max_bounds - min_bounds))
    return center, scene_size


def parse_frame_index(pcd_name):
    """Parse the numeric frame index from names like pcd_00029."""
    try:
        return int(pcd_name.split('_')[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid frame name: {pcd_name}") from exc


def load_camera_pose_files(pcd_root, pcd_scene, pcd_name):
    """Load camera extrinsic and intrinsic for a specific frame."""
    frame_idx = parse_frame_index(pcd_name)
    pose_dir = Path(pcd_root) / "pose" / pcd_scene
    cam2world_path = pose_dir / f"cam2world_{frame_idx:05d}.txt"
    intrinsic_path = pose_dir / f"intrinsic_{frame_idx:05d}.txt"

    if not cam2world_path.exists():
        raise FileNotFoundError(f"Missing camera extrinsic: {cam2world_path}")
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Missing camera intrinsic: {intrinsic_path}")

    cam2world = np.loadtxt(cam2world_path)
    intrinsic = np.loadtxt(intrinsic_path)
    return cam2world, intrinsic, cam2world_path, intrinsic_path


def draw_line_segment(start,
                      end,
                      color=(1.0, 0.0, 0.0),
                      line_width=4.0,
                      tube_radius=0.03):
    xs = [start[0], end[0]]
    ys = [start[1], end[1]]
    zs = [start[2], end[2]]
    mlab.plot3d(xs,
                ys,
                zs,
                color=color,
                tube_radius=tube_radius,
                line_width=line_width,
                reset_zoom=False)


def compute_frustum_corners(cam2world, intrinsic, scene_size):
    """Compute frustum corners in world coordinates for camera fitting/drawing."""
    origin = cam2world[:3, 3]
    rotation = cam2world[:3, :3]
    inv_intrinsic = np.linalg.inv(intrinsic)

    image_width = int(round(intrinsic[0, 2] * 2))
    image_height = int(round(intrinsic[1, 2] * 2))
    image_corners = np.array([
        [0.0, 0.0, 1.0],
        [image_width - 1.0, 0.0, 1.0],
        [image_width - 1.0, image_height - 1.0, 1.0],
        [0.0, image_height - 1.0, 1.0],
    ])

    frustum_length = max(scene_size * 0.35, 0.5)
    frustum_corners = []
    for corner in image_corners:
        ray_cam = inv_intrinsic @ corner
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = rotation @ ray_cam
        frustum_corners.append(origin + ray_world * frustum_length)

    return origin, np.asarray(
        frustum_corners), frustum_length, image_width, image_height


def draw_camera_frustum(cam2world,
                        intrinsic,
                        scene_size,
                        color=(1.0, 0.1, 0.1)):
    """Draw a camera frustum using cam2world and intrinsic matrices."""
    if cam2world.shape != (4, 4):
        raise ValueError(f"cam2world must be 4x4, got {cam2world.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must be 3x3, got {intrinsic.shape}")

    origin, frustum_corners, frustum_length, image_width, image_height = compute_frustum_corners(
        cam2world, intrinsic, scene_size)

    mlab.points3d(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        scale_factor=max(scene_size * 0.03, 0.05),
        mode="sphere",
        color=color,
        opacity=1.0,
        reset_zoom=False,
    )

    # frustum_tube_radius = max(scene_size * 0.01, 0.025)
    frustum_tube_radius = 0.03
    frustum_line_width = 1.0

    for corner in frustum_corners:
        draw_line_segment(
            origin,
            corner,
            color=color,
            line_width=frustum_line_width,
            tube_radius=frustum_tube_radius,
        )

    for idx in range(4):
        draw_line_segment(
            frustum_corners[idx],
            frustum_corners[(idx + 1) % 4],
            color=color,
            line_width=frustum_line_width,
            tube_radius=frustum_tube_radius,
        )

    print(
        f"[INFO] Camera frustum drawn with image size {image_width}x{image_height} and length {frustum_length:.2f}"
    )


def get_camera_params(config, scene_name):
    default_params = {
        'azimuth': 0,
        'elevation': 0,
        'parallel_scale_factor': 0.6,
        'center_offset': [0.0, 0.0, 0.0],
        'image_size': [1600, 1600]
    }

    if config is None:
        return default_params

    if scene_name in config:
        scene_params = config[scene_name].copy()
        print(f"[INFO] Using camera config for {scene_name}")
        scene_params.setdefault('azimuth', 0)
        scene_params.setdefault('elevation', 0)
        scene_params.setdefault('parallel_scale_factor', 0.6)
        scene_params.setdefault('center_offset', [0.0, 0.0, 0.0])
        scene_params.setdefault('image_size', [1600, 1600])
        return scene_params

    print(
        f"[INFO] No specific config for {scene_name}, using default top-down view"
    )
    return config.get('default', default_params)


def setup_camera_view(voxel_centers, camera_params):
    min_bounds = np.min(voxel_centers, axis=0)
    max_bounds = np.max(voxel_centers, axis=0)
    center = (min_bounds + max_bounds) / 2
    scene_size = np.max(max_bounds - min_bounds)

    fixed_bounds = camera_params.get('fixed_bounds')
    if isinstance(
            fixed_bounds, dict
    ) and 'min_bounds' in fixed_bounds and 'max_bounds' in fixed_bounds:
        fixed_min = np.array(fixed_bounds['min_bounds'], dtype=float)
        fixed_max = np.array(fixed_bounds['max_bounds'], dtype=float)
        if fixed_min.shape == (3, ) and fixed_max.shape == (3, ):
            min_bounds = fixed_min
            max_bounds = fixed_max
            center = (min_bounds + max_bounds) / 2
            scene_size = np.max(max_bounds - min_bounds)
            print("[INFO] Camera framing mode: fixed_bounds")
    elif 'fixed_center' in camera_params and 'fixed_scene_size' in camera_params:
        fixed_center = np.array(camera_params['fixed_center'], dtype=float)
        fixed_scene_size = float(camera_params['fixed_scene_size'])
        if fixed_center.shape == (3, ) and fixed_scene_size > 0:
            center = fixed_center
            scene_size = fixed_scene_size
            print("[INFO] Camera framing mode: fixed_center_size")
    elif 'fixed_scene_size' in camera_params:
        fixed_scene_size = float(camera_params['fixed_scene_size'])
        if fixed_scene_size > 0:
            scene_size = fixed_scene_size
            print("[INFO] Camera framing mode: fixed_scene_size")
    else:
        print("[INFO] Camera framing mode: dynamic")

    center_offset = camera_params.get('center_offset', [0.0, 0.0, 0.0])
    center[0] += center_offset[0]
    center[1] += center_offset[1]
    center[2] += center_offset[2]

    print(
        f"[INFO] Scene center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
    )
    print(f"[INFO] Scene size: {scene_size:.2f}")

    mlab.gcf().scene.camera.parallel_projection = True
    print("[INFO] Parallel projection enabled (fixed top-down view)")

    azimuth = camera_params.get('azimuth', 0)
    elevation = camera_params.get('elevation', 0)
    parallel_scale_factor = camera_params.get('parallel_scale_factor', 0.6)

    mlab.view(azimuth=azimuth, elevation=elevation, focalpoint=center)
    parallel_scale = scene_size * parallel_scale_factor
    mlab.gcf().scene.camera.parallel_scale = parallel_scale

    print(
        f"[INFO] Camera view: azimuth={azimuth}, elevation={elevation}, parallel_scale={parallel_scale:.2f}"
    )


def visualize_voxels_with_original_colors(pcd_path,
                                          voxel_size=0.08,
                                          show_3d=True,
                                          save_image=False,
                                          output_path=None,
                                          camera_params=None,
                                          image_size=None,
                                          include_frustum=False,
                                          cam2world=None,
                                          intrinsic=None):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print(f"[INFO] Loaded {points.shape[0]} points from {pcd_path}")

    colors = (colors * 255).astype(np.uint8)
    voxel_coords = np.floor((points / voxel_size) + 1e-4).astype(int)

    voxel_dict = {}
    for idx, voxel in enumerate(voxel_coords):
        key = tuple(voxel)
        if key not in voxel_dict:
            voxel_dict[key] = {'colors': [colors[idx]], 'count': 1}
        else:
            voxel_dict[key]['colors'].append(colors[idx])
            voxel_dict[key]['count'] += 1

    if image_size is None:
        image_size = [1600, 1600]

    mlab.figure(size=tuple(image_size), bgcolor=(1.0, 1.0, 1.0))
    print(f"[INFO] Figure size: {image_size[0]}x{image_size[1]}")

    color_groups = {}
    all_voxel_centers = []

    for voxel_idx, data in voxel_dict.items():
        avg_color = np.mean(data['colors'], axis=0).astype(int)
        color_key = tuple(avg_color)

        if color_key not in color_groups:
            color_groups[color_key] = []

        center = np.array(voxel_idx) * voxel_size + voxel_size / 2
        color_groups[color_key].append(center)
        all_voxel_centers.append(center)

    all_voxel_centers = np.array(all_voxel_centers)
    if all_voxel_centers.size > 0:
        actual_center = np.mean(all_voxel_centers, axis=0)
        print(
            f"[INFO] Actual voxel center: [{actual_center[0]:.2f}, {actual_center[1]:.2f}, {actual_center[2]:.2f}]"
        )

        scene_min = np.min(all_voxel_centers, axis=0)
        scene_max = np.max(all_voxel_centers, axis=0)
        scene_size = float(np.max(scene_max - scene_min))
    else:
        actual_center = np.zeros(3)
        scene_size = 1.0
        print("[WARNING] No voxel centers found for this frame")

    for color_rgb, centers in color_groups.items():
        centers = np.array(centers)
        color_normalized = np.array(color_rgb) / 255.0

        mlab.points3d(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            scale_factor=voxel_size * 1.0,
            mode="cube",
            color=tuple(color_normalized),
            opacity=1.0,
            resolution=8,
        )

    print(
        f"[INFO] Rendered {len(all_voxel_centers)} voxels in {len(color_groups)} color groups"
    )

    if camera_params is None:
        camera_params = {
            'azimuth': 0,
            'elevation': 0,
            'parallel_scale_factor': 0.6,
            'center_offset': [0.0, 0.0, 0.0],
            'image_size': [1600, 1600]
        }

    camera_fit_centers = all_voxel_centers
    if include_frustum and cam2world is not None and intrinsic is not None:
        fit_scene_size = scene_size
        if camera_params is not None and 'fixed_scene_size' in camera_params:
            fit_scene_size = float(camera_params['fixed_scene_size'])
        frustum_origin, frustum_corners, _, _, _ = compute_frustum_corners(
            cam2world, intrinsic, fit_scene_size)

        if camera_fit_centers.size > 0:
            camera_fit_centers = np.vstack([
                camera_fit_centers,
                frustum_origin.reshape(1, 3),
                frustum_corners,
            ])
        else:
            camera_fit_centers = np.vstack([
                frustum_origin.reshape(1, 3),
                frustum_corners,
            ])

    setup_camera_view(camera_fit_centers, camera_params)

    if include_frustum:
        if cam2world is None or intrinsic is None:
            print(
                "[WARNING] Frustum requested but camera pose files are unavailable"
            )
        else:
            frustum_scene_size = scene_size
            if camera_params is not None and 'fixed_scene_size' in camera_params:
                frustum_scene_size = float(camera_params['fixed_scene_size'])
            draw_camera_frustum(cam2world, intrinsic, frustum_scene_size)

    if save_image and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlab.savefig(str(output_path), size=tuple(image_size))
        remove_white_background(str(output_path))
        print(f"[INFO] Image saved to: {output_path}")

    if show_3d:
        print("[INFO] Showing 3D interactive interface...")
        mlab.show()
    else:
        print("[INFO] Skipping 3D interface, only saving image...")


if __name__ == "__main__":
    if len(sys.argv) != 11:
        print(
            "Usage: python3 vis_occ_online.py <pcd_root> <pcd_fold> <pcd_scene> <pcd_name> <pcd_ext> <output_folder> <show_3d> <include_frustum> <use_last_frame_baseline> <is_local_mode>"
        )
        sys.exit(1)

    pcd_root = Path(sys.argv[1])
    pcd_fold = sys.argv[2]
    pcd_scene = sys.argv[3]
    pcd_name = sys.argv[4]
    pcd_ext = sys.argv[5]
    output_folder = sys.argv[6]
    show_3d = sys.argv[7].lower() in ['true', '1', 'yes', 'y']
    include_frustum = sys.argv[8].lower() in ['true', '1', 'yes', 'y']
    use_last_frame_baseline = sys.argv[9].lower() in ['true', '1', 'yes', 'y']
    is_local_mode = sys.argv[10].lower() in ['true', '1', 'yes', 'y']

    pcd_file = pcd_root / pcd_fold / pcd_scene / (pcd_name + pcd_ext)

    config_path = Path(__file__).parent / "camera_config_online.json"
    camera_config = load_camera_config(config_path)
    camera_params = get_camera_params(camera_config, pcd_scene)

    # Default scale reference: last frame of the same scene.
    reference_pcd_name = "pcd_00029"
    reference_pcd_path = pcd_root / pcd_fold / pcd_scene / (
        reference_pcd_name + pcd_ext)
    if use_last_frame_baseline:
        try:
            fixed_center, fixed_scene_size = estimate_scene_center_and_size_from_pcd(
                reference_pcd_path, voxel_size=0.08)
            camera_params['fixed_center'] = fixed_center.tolist()
            camera_params['fixed_scene_size'] = fixed_scene_size
            print(
                f"[INFO] Fixed scene center from last frame ({reference_pcd_name}): [{fixed_center[0]:.2f}, {fixed_center[1]:.2f}, {fixed_center[2]:.2f}]"
            )
            print(
                f"[INFO] Fixed scene size from last frame ({reference_pcd_name}): {fixed_scene_size:.2f}"
            )
        except Exception as exc:
            print(
                f"[ERROR] Last-frame baseline is enabled but failed to load baseline from {reference_pcd_path}: {exc}"
            )
            sys.exit(1)
    elif not use_last_frame_baseline:
        # Force adaptive framing when baseline is disabled.
        camera_params.pop('fixed_bounds', None)
        camera_params.pop('fixed_center', None)
        camera_params.pop('fixed_scene_size', None)
        print(
            "[INFO] Last-frame baseline disabled; forcing per-frame adaptive framing"
        )

    image_size = camera_params.get('image_size', [1600, 1600])

    cam2world = None
    intrinsic = None
    if include_frustum:
        try:
            cam2world, intrinsic, cam2world_path, intrinsic_path = load_camera_pose_files(
                pcd_root, pcd_scene, pcd_name)
            print(f"[INFO] Loaded camera pose: {cam2world_path}")
            print(f"[INFO] Loaded camera intrinsics: {intrinsic_path}")
        except Exception as exc:
            print(f"[WARNING] Unable to load camera pose for frustum: {exc}")
            include_frustum = False

    output_dir = pcd_root / output_folder / pcd_scene
    output_suffix = ""
    if is_local_mode:
        output_suffix += "_local"
    output_suffix += "_frustum" if include_frustum else "_nofrustum"
    output_path = output_dir / (pcd_name + output_suffix + ".png")

    visualize_voxels_with_original_colors(
        pcd_file,
        voxel_size=0.08,
        show_3d=show_3d,
        save_image=True,
        output_path=output_path,
        camera_params=camera_params,
        image_size=image_size,
        include_frustum=include_frustum,
        cam2world=cam2world,
        intrinsic=intrinsic,
    )
