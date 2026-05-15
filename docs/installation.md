# Installation

## 1. Create conda environment
```bash
conda create -n VEOcc python=3.8.19
conda activate VEOcc
```

## 2. Install PyTorch
```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
```

## 3. Install some packages

### 1. Install packages from MMLab
```bash
pip install openmim==0.3.9
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.2.2
mim install mmdet3d==1.1.1
```

### 2. Install other packages
```bash
# pip install spconv-cu114==2.3.6
pip install timm
pip install vtk==9.0.1
```

### 3. Install custom CUDA ops
```bash
cd model/lifter/bev_pool && python setup.py build_ext --inplace
cd ../../..

# We compile bev_pool with cuda 11.8

# (VEOcc) ➜  VEOcc-Offical git:(master) ✗ nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Wed_Sep_21_10:33:58_PDT_2022
# Cuda compilation tools, release 11.8, V11.8.89
# Build cuda_11.8.r11.8/compiler.31833905_0
```

## 4. Install the additional dependencies
```bash
pip install -r requirements.txt
```

## 5. Download Depth-Anything-V2 and make some slight changes
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
mv Depth-Anything-V2 Depth_Anything_V2
```

**Folder structure**
```
VEOcc
├── ...
├── Depth_Anything_V2
```

Go to **Depth_Anything_V2/metric_depth/depth_anything_v2/dpt.py** and change the function **infer_image** in the class **DepthAnythingV2** as follows:
```Python
def infer_image(self, image, h_, w_, input_size=518):
    depth = self.forward(image)
    depth = F.interpolate(depth[:, None], (h_, w_), mode="bilinear", align_corners=True)[0, 0]
    return depth
```


## 7. Download [finetuned checkpoint](https://huggingface.co/YkiWu/EmbodiedOcc) of Depth-Anything-V2 on Occ-ScanNet and put it under the **checkpoints**

**Folder structure**
```
VEOcc
├── ...
├── checkpoints/
│   ├── finetune_scannet_depthanythingv2.pth
```
