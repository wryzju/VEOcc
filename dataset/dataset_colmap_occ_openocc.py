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
class Colmap_Scene_OpenOccupancy_Dataset_OffLine_Depth(data.Dataset):

    def __init__(
        self,
        scene_name='csc105_2floor_260513',
        num_frames=1,
        offset=0,
        grid_size_occ=[60, 60, 36],
        phase='train',
    ):
        self.root = 'data/colmap_made'
        self.scene_name = scene_name
        self.num_frames = num_frames
        self.offset = offset
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
        return len(self.image_paths)

    def __getitem__(self, index):
        rgb_path = self.image_paths[index]
        img_idx = rgb_path.split("/")[-1].split(".")[0]

        meta = {}
        meta['name'] = f"{self.scene_name}/{img_idx}"
        meta['sample_index'] = int(index)
        meta['prev_index'] = -1
        meta['scene_size'] = self.scene_size

        # Read camera pose
        pose_path = rgb_path.replace('.jpg', '.txt')
        if os.path.exists(pose_path):
            cam_pose = np.loadtxt(pose_path)
        else:
            cam_pose = np.eye(4)

        meta['cam2world'] = cam_pose
        world2cam = np.linalg.inv(cam_pose)
        meta['world2cam'] = world2cam

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

        img_depthbranch = cv2.imread(rgb_path)
        img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
        img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0

        depth_gt_np = np.ones((480, 640), dtype=np.float32)  # dummy
        depth_pred_np = np.ones((480, 640), dtype=np.float32)  # dummy

        sample = transform({'image': img_depthbranch, 'depth': depth_gt_np})
        img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
        depth_gt_np = torch.from_numpy(depth_gt_np).unsqueeze(0)
        depth_pred_np = torch.from_numpy(depth_pred_np).to(torch.float32)

        meta['depth_gt_np'] = depth_gt_np
        meta['depth_pred_np'] = depth_pred_np

        depth_valid_mask = (torch.isnan(depth_gt_np) == 0)
        depth_gt_np[depth_valid_mask == 0] = 0
        meta['img_depthbranch'] = img_depthbranch
        meta['depth_gt_np_valid'] = depth_gt_np

        meta['rgb_path'] = rgb_path

        N_img = []
        this_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        this_H, this_W = this_img.shape[:2]
        new_H, new_W = 480, 640
        new_img = cv2.resize(this_img, (new_W, new_H))
        W_factor = new_W / this_W
        H_factor = new_H / this_H
        N_img.append(new_img)
        img = np.stack(N_img, 0)
        img = [img]

        cam_intrin = self.cam_k.copy()
        cam_intrin[0, 0] *= W_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[1, 2] *= H_factor

        meta['cam_k'] = cam_intrin
        viewpad = np.eye(4)
        viewpad[:3, :3] = cam_intrin
        meta['cam2img'] = viewpad
        meta['world2img'] = viewpad @ world2cam

        # cam_loc = cam_pose[:3, 3]
        # meta['vox_origin'] = np.round(cam_loc - np.array([2.4, 2.4, 1.4]), 4)

        cam_loc = cam_pose[:3, 3]
        look_forward_dir = cam_pose[:3, 2]
        # 我们把盒子中心顺着视线往前推场景深度的 35% ~ 40%（留一点空间包裹机身盲区）
        push_distance = 4.8 * 0.4  # 你也可以改写为动态的 (self.scene_size[0] * 0.4)
        box_center_world = cam_loc + look_forward_dir * push_distance
        meta['vox_origin'] = np.round(box_center_world - np.array([2.4, 2.4, 2.0]), 4)

        occ = np.ones(self.grid_size_occ, dtype=np.uint8) * 12

        projected_pix, fov_mask, pix_z, occ_xyz = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size,
            new_W,
            new_H,
            self.scene_size,
            dim_60_60_36=True,
        )

        meta['projected_pix'] = projected_pix
        meta['fov_mask'] = fov_mask.reshape(*self.grid_size_occ)
        meta['pix_z'] = pix_z
        meta['occ_xyz'] = occ_xyz.reshape(*self.grid_size_occ, 3)

        vox_near = meta['vox_origin']
        vox_far = vox_near + self.scene_size
        meta['nyu_pc_range'] = np.concatenate([vox_near, vox_far], axis=0)

        cam_vox_near = np.array([-5, -6, -3])
        cam_vox_far = np.array([5, 6, 8])
        meta['cam_vox_range'] = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)

        meta['occ_mask_valid'] = (occ != 0)
        meta['label'] = occ

        occ = [occ]
        imgs = np.stack(img, 0)
        occs = np.stack(occ, 0)

        return (imgs, meta, occs)
