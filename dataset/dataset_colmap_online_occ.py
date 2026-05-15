import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import cv2
from torchvision.transforms import Compose
from dataset.transform_ import Resize, NormalizeImage, PrepareForNet
from . import OPENOCC_DATASET
from dataset.nyu_utils import vox2pix


@OPENOCC_DATASET.register_module()
class Colmap_Online_SceneOcc_Dataset(data.Dataset):

    def __init__(
        self,
        scene_name='csc105_2floor_260513',
        num_frames=30,
        grid_size_occ=[60, 60, 36],
        phase='train',
    ):
        self.root = 'data/colmap_made'
        self.scene_name = scene_name
        self.num_frames = num_frames
        self.grid_size_occ = grid_size_occ
        self.phase = phase
        self.voxel_size = 0.08
        self.scene_size = (4.8, 4.8, 2.88)

        self.scene_dir = os.path.join(self.root, self.scene_name)
        self.posed_images_dir = os.path.join(self.scene_dir, 'posed_images')

        self.image_paths = sorted(
            [os.path.join(self.posed_images_dir, f) for f in os.listdir(self.posed_images_dir) if f.endswith('.jpg')])

        # Parse intrinsics
        self.intrinsics_path = os.path.join(self.scene_dir, 'camera_intrinsics.txt')
        self.cam_k = np.eye(3)
        if os.path.exists(self.intrinsics_path):
            with open(self.intrinsics_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('FX'):
                        self.cam_k[0, 0] = float(line.split()[1])
                    elif line.startswith('FY'):
                        self.cam_k[1, 1] = float(line.split()[1])
                    elif line.startswith('CX'):
                        self.cam_k[0, 2] = float(line.split()[1])
                    elif line.startswith('CY'):
                        self.cam_k[1, 2] = float(line.split()[1])

    def __len__(self):
        # The online dataset usually iterates over subscenes/scenes.
        # Here we have just one scene, so length is 1 or we can break it to chunks.
        # Let's say we have 1 scene.
        return 1

    def __getitem__(self, index):
        meta = {}
        meta['scene_name'] = self.scene_name
        meta['valid_img_paths'] = self.image_paths

        # Determine global_scene_origin from first frame
        if len(self.image_paths) > 0:
            first_frame_path = self.image_paths[0]
            pose_path = first_frame_path.replace('.jpg', '.txt')
            if os.path.exists(pose_path):
                first_cam_pose = np.loadtxt(pose_path)
            else:
                first_cam_pose = np.eye(4)
            first_cam_loc = first_cam_pose[:3, 3]
            look_forward_dir = first_cam_pose[:3, 2]
            push_distance = 4.8 * 0.4
            box_center_world = first_cam_loc + look_forward_dir * push_distance
            first_vox_origin = box_center_world - np.array([2.4, 2.4, 2.0])
            meta['global_scene_origin'] = np.round(first_vox_origin, 4)
        else:
            meta['global_scene_origin'] = np.array([-5.0, -5.0, -2.0])

        meta['global_scene_dim'] = [100, 100, 50]
        meta['global_scene_size'] = 0.08 * np.array([100, 100, 50])
        meta['global_pts'] = np.zeros((100, 100, 50, 3), dtype=np.float32)  # dummy
        meta['global_labels'] = np.ones((100, 100, 50), dtype=np.uint8) * 12  # dummy empty
        meta['global_mask'] = np.ones((100, 100, 50), dtype=np.bool_)

        transform = Compose([
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        monometa_list = []
        N_img = []
        N_occ = []

        # For simplicity in this dummy version, we take all or limited frames
        frames_to_load = self.image_paths[:self.num_frames]
        if len(frames_to_load) == 0:
            frames_to_load = self.image_paths

        for rgb_path in frames_to_load:
            monometa = {}
            monometa['global_scene_origin'] = meta['global_scene_origin']
            monometa['global_scene_size'] = meta['global_scene_size']

            img_idx = rgb_path.split("/")[-1].split(".")[0]
            monometa['name'] = f"{self.scene_name}/{img_idx}"

            # Read camera pose
            pose_path = rgb_path.replace('.jpg', '.txt')
            if os.path.exists(pose_path):
                cam_pose = np.loadtxt(pose_path)
            else:
                cam_pose = np.eye(4)

            monometa['cam2world'] = cam_pose
            world2cam = np.linalg.inv(cam_pose)
            monometa['world2cam'] = world2cam

            img_depthbranch = cv2.imread(rgb_path)
            img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
            img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
            sample = transform({'image': img_depthbranch})
            img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
            monometa['img_depthbranch'] = img_depthbranch
            monometa['rgb_path'] = rgb_path
            monometa['depth_path'] = rgb_path  # dummy
            monometa['scene_size'] = self.scene_size  # added missing scene_size

            monometa['depth_pred_np'] = torch.ones((1, 480, 640))  # dummy

            this_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            this_H, this_W = this_img.shape[:2]
            new_H, new_W = 480, 640
            new_img = cv2.resize(this_img, (new_W, new_H))
            W_factor = new_W / this_W
            H_factor = new_H / this_H
            N_img.append(new_img)

            cam_intrin = self.cam_k.copy()
            cam_intrin[0, 0] *= W_factor
            cam_intrin[0, 2] *= W_factor
            cam_intrin[1, 1] *= H_factor
            cam_intrin[1, 2] *= H_factor

            monometa['cam_k'] = cam_intrin
            viewpad = np.eye(4)
            viewpad[:3, :3] = cam_intrin
            monometa['cam2img'] = viewpad
            monometa['world2img'] = viewpad @ world2cam

            monometa['depth_gt'] = np.ones((new_H, new_W))  # dummy

            # Set voxel origin based on cam location to have local volume
            cam_loc = cam_pose[:3, 3]
            look_forward_dir = cam_pose[:3, 2]
            push_distance = 4.8 * 0.4  # 你也可以改写为动态的 (self.scene_size[0] * 0.4)
            box_center_world = cam_loc + look_forward_dir * push_distance
            raw_vox_origin = box_center_world - np.array([2.4, 2.4, 1.4])

            # Align with global_scene_origin
            offset = raw_vox_origin - meta['global_scene_origin']
            offset_n = np.round(offset / self.voxel_size)
            monometa['vox_origin'] = np.round(meta['global_scene_origin'] + offset_n * self.voxel_size, 4)

            occ = np.ones(self.grid_size_occ, dtype=np.uint8) * 12  # empty space
            N_occ.append(occ)

            projected_pix, fov_mask, pix_z, occ_xyz = vox2pix(
                world2cam,
                monometa['cam_k'],
                monometa['vox_origin'],
                self.voxel_size,
                new_W,
                new_H,
                self.scene_size,
                dim_60_60_36=True,
            )

            monometa['projected_pix'] = projected_pix
            monometa['fov_mask'] = fov_mask.reshape(*self.grid_size_occ)
            monometa['pix_z'] = pix_z
            monometa['occ_xyz'] = occ_xyz.reshape(*self.grid_size_occ, 3)

            vox_near = monometa['vox_origin']
            vox_far = vox_near + self.scene_size
            monometa['nyu_pc_range'] = np.concatenate([vox_near, vox_far], axis=0)

            cam_vox_near = np.array([-5, -6, -3])
            cam_vox_far = np.array([5, 6, 8])
            monometa['cam_vox_range'] = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)

            monometa['occ_mask_valid'] = (occ != 0)
            monometa['label'] = occ
            monometa['mask_in_global_from_this'] = np.ones(self.grid_size_occ, dtype=np.bool_)  # dummy

            monometa_list.append(monometa)

        meta['monometa_list'] = monometa_list

        imgs = np.stack(N_img, 0)
        imgs = np.expand_dims(imgs, axis=0)
        occs = np.stack(N_occ, 0)

        return (imgs, meta, occs)
