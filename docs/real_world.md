# Real-World Data Preparation

This repository provides a small preprocessing pipeline that turns raw videos into COLMAP sparse reconstruction results and then generates the `posed_images` required for inference.

## Environment Setup
```bash
conda activate VEOcc
conda install -c conda-forge colmap -y
```

## Data Preparation

Prepare a data directory that contains these two files:

- `video.mp4` : Shoot a video with your mobile phone. It is recommended to use a camera app with fixed‑focus functionality.
- `calib.mp4` : A calibration video of a chessboard captured from multiple viewpoints

We provided example data in `example/hotel_260516`.

## Workflow

### Step 1: Preprocess

Extract frames, crop them, and export camera intrinsics.

```bash
python scripts/preprocess.py example/hotel_260516 --fps 10
```
After preprocessing, the following folders and files will be created:

- `images/` : network input images
- `images_colmap/` : COLMAP input images
- `calib_frames/`
- `calib_frames_colmap/`
- `camera_intrinsics.txt`
- `camera_intrinsics_colmap.txt`

### Step 2: Run Sparse Reconstruction in the COLMAP GUI

Open the COLMAP GUI and use `images_colmap/` for feature extraction, matching, and sparse reconstruction.
```bash
colmap gui
```

Recommended flow:

1. `File/New project`, select `images_colmap/` as image folder
2. `Processing/Feature Extraction` and `Processing/Feature Matching`, we use exhaustive matching in single-room sequence and use sequential matching in multi-room sequences.
4. `Reconstruction/Start reconstruction`, this step takes some time.
5. `File/Save model as text`, export model as text files into `data_dir/sparse_txt/`
6. Calculate the scale factor by dividing the real-world distance between two SIFT feature points by their corresponding distance in the COLMAP coordinate system. You can retrieve the coordinates by double-clicking points in the GUI. Since the scale factor varies across individual reconstructions, the scale factor we provide may not be accurate.

Make sure the following file exists:

```text
example/hotel_260516/sparse_txt/images.txt
```

### Step 3: Generate `posed_images` for Inference

Generate posed images and the corresponding 4x4 pose files from the COLMAP sparse output.

```bash
python scripts/generate_posed_images.py example/hotel_260516 --scale_factor 0.425 --interval 5 --start_frame 100 --end_frame 630 --rx -30 --ry 0 --rz 0

# Set --start_frame and --end_frame to filter out useless frames
# Use --rx, --ry and --rz to apply global rotation to all camera poses.
```

Output will be written to:

- `example/hotel_265016/posed_images/00000.jpg`
- `example/hotel_265016/posed_images/00000.txt`
- `example/hotel_265016/posed_images/camera_intrinsics.txt`

### Step4: Modify evaluation configs
Softlink your data path to `data/colmap_made`.

```
ln -s example/hotel_260516 data/colmap_made/hotel_260516_example
```
Then change scene name in `config/eval_embodied_colmap.py`.
```
# scene_name = 'hotel_260516'
# scene_name = 'household_1_260523'
# scene_name = 'household_2_260523'

scene_name = 'hotel_260516_example' # change here

num_frames = 0 # for all frames

train_dataset_config = dict(
    type='Colmap_Online_SceneOcc_Dataset',
    scene_name=scene_name,
    num_frames=num_frames,
    phase='train',
)

val_dataset_config = dict(
    type='Colmap_Online_SceneOcc_Dataset',
    scene_name=scene_name,
    num_frames=num_frames,
    phase='test',
)
```

### Step5: Run and visualize
```
torchrun --nproc_per_node=1 eval_embodied_colmap.py --py-config config/eval_embodied_colmap.py --work-dir workdir/eval_embodied_colmap --ckpt-path checkpoints/embodied.pth --save-ply

bash vis_occ_online.sh
```
