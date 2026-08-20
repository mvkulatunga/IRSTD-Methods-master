#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add VMamba to system path for SS2D imports
vmamba_path = "/home/thor/Programming/IRSTD/methods/VMamba"
if vmamba_path not in sys.path:
    sys.path.append(vmamba_path)

# Import the core 2D Selective Scan backbone
try:
    from classification.models.vmamba import SS2D
except ImportError as e:
    print("Ensure the VMamba repository is properly compiled at the absolute path.")
    raise e


# The config is pulled from SSTnet, the MOCID config follows SSTnet

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import os

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")


class Config:
    T = 5
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4

    LR_INIT = 0.01
    MIN_LR = 1e-4
    WARMUP_EPOCHS = 4
    MOMENTUM = 0.937
    WEIGHT_DECAY = 5e-4
    EPOCHS_SPTBACKBONE = 100
    EPOCHS_DAM = 100
    EVAL_EVERY = 4

    train_path = "../../datasets/DAUB/train_DAUB.txt"
    val_path = "../../datasets/DAUB/val_DAUB.txt"


# 1. data augmentation: random flipping of video clips
# 2. clips batched with T = 5, with T - 1 reference clips
# 3. clip starts with 5th frame, since we want data to have spatio-temporal consistency

# In[3]:


import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class MOCIDDataset(Dataset):
    def __init__(self, annotations_file, T=5, img_size=(512, 512), is_train=True):
        """Builds T-frame clips from an annotation file, where the last frame
        in each clip carries the target bounding box and class label.
        (Works with both, DAUB and IRSDT-15k)

        Args:
            annotations_file (str): Path to the annotation file. Each line must
                be formatted as "<img_path> <xmin,ymin,xmax,ymax,class_id>".
            T (int): Clip length (number of consecutive frames per sample). Defaults to 5.
            img_size (tuple): Output (width, height) that every frame and its
                bounding box are resized to. Defaults to (512, 512).
            is_train (bool): If True, enables train-time augmentation (random
                horizontal flip). Defaults to True.
        """

        self.T = T  # clip length
        self.img_size = img_size

        # for train/eval flag
        self.is_train = is_train
        self.clips = []

        sequences = defaultdict(list)
        if not os.path.exists(annotations_file):
            print(f"Warning: {annotations_file} not found.")
            return

        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) != 2:
                    continue
                img_path, bbox_str = parts[0], parts[1]
                seq_id = os.path.basename(os.path.dirname(img_path))
                frame_num = int(os.path.basename(img_path).split(".")[0])
                box_data = list(map(int, bbox_str.split(",")))
                sequences[seq_id].append(
                    {
                        "path": img_path,
                        "frame_num": frame_num,
                        "bbox": box_data[:4],
                        "class_id": box_data[4],
                    }
                )

        for seq_id, frames in sequences.items():
            # Sort frames chronologically (based on frame number)
            frames = sorted(frames, key=lambda x: x["frame_num"])

            # START FROM T-1 (e.g., 4) to ensure we always have 5 distinct frames
            for i in range(self.T - 1, len(frames)):
                # Grab the current frame and the T-1 frames before it
                idxs = [i - (self.T - 1) + j for j in range(self.T)]
                self.clips.append([frames[k] for k in idxs])

        print(
            f"Loaded {len(self.clips)} clips of length {self.T} from {annotations_file}"
        )

    def __len__(self):
        """Returns the total number of clips in the dataset.

        Returns:
            int: Number of T-frame clips available.
        """
        return len(self.clips)

    def __getitem__(self, idx):
        """Loads and preprocesses a single clip: reads T frames, resizes them
        to `img_size`, in train mode applies a synchronized horizontal flip randomly,
        and rescales the target frame's bounding box to
        match the resized image.

        Args:
            idx (int): Index of the clip to fetch.

        Returns:
            tuple:
                imgs_tensor (torch.FloatTensor): Shape (T, 3, H, W), RGB frames
                    normalized to [0, 1].
                target (dict): Contains
                    "boxes" (torch.FloatTensor): Shape (1, 4), [xmin, ymin, xmax, ymax]
                        in resized image coordinates.
                    "labels" (torch.LongTensor): Shape (1,), class id of the target frame.
        """
        clip_data = self.clips[idx]
        imgs = []

        target_info = clip_data[-1]
        orig_xmin, orig_ymin, orig_xmax, orig_ymax = target_info["bbox"]
        target_class = target_info["class_id"]

        sample_img = cv2.imread(clip_data[0]["path"], cv2.IMREAD_COLOR)
        if sample_img is None:
            raise FileNotFoundError(f"Could not read image at {clip_data[0]['path']}")

        orig_h, orig_w = sample_img.shape[:2]
        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h

        xmin = int(orig_xmin * scale_x)
        ymin = int(orig_ymin * scale_y)
        xmax = int(orig_xmax * scale_x)
        ymax = int(orig_ymax * scale_y)

        do_flip = self.is_train and (random.random() > 0.5)
        if do_flip:
            xmin, xmax = self.img_size[0] - xmax, self.img_size[0] - xmin

        for frame in clip_data:
            img = cv2.imread(frame["path"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            if do_flip:
                img = cv2.flip(img, 1)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)

        imgs_tensor = torch.tensor(np.array(imgs), dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32),
            "labels": torch.tensor([target_class], dtype=torch.int64),
        }
        return imgs_tensor, target


def collate_mocid(batch):
    """Collate function for MOCIDDataset that stacks clips into a batch and
    converts each target's bounding box into (cx, cy, w, h, class_id) format.

    Args:
        batch (list): List of (imgs_tensor, target) tuples as returned by
            MOCIDDataset.__getitem__.

    Returns:
        tuple:
            clips (torch.FloatTensor): Shape (B, T, 3, H, W), stacked clips.
            labels (list[torch.FloatTensor]): Per-sample tensors of shape
                (num_boxes, 5) holding [cx, cy, w, h, class_id].
    """
    clips = torch.stack([b[0] for b in batch])
    labels = []
    for _, tgt in batch:
        bx = tgt["boxes"]
        cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2
        w, h = bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]
        cls = tgt["labels"].float()
        labels.append(torch.stack([cx, cy, w, h, cls], dim=1))
    return clips, labels


# <img src="./diagrams/csplayer_architecture.svg" alt="csplayer_architecture" style="max-width: 50%; max-height: 400px">
# 
# 
# CSPLayer Architecture, as used in [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)

# In[4]:


from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# CSPnet convolution block
class BaseConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# Bottleneck, also referred to as
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks=1):
        super().__init__()
        hidden_channels = out_channels // 2

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1)

        self.bottlenecks = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels)
                for _ in range(num_bottlenecks)
            ]
        )

        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1)

    def forward(self, x):
        # main path
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)

        # bypass path
        x_2 = self.conv2(x)

        # concatenate
        out = torch.cat((x_1, x_2), dim=1)
        return self.conv3(out)


# <img src="./diagrams/spatialfista_architecture.svg" alt="spatialfista_architecture" style="max-width: 50%; max-height: 400px">
# 
# 
# SpatialFISTA processes the input feature $f \in \mathbb{R}^{T \times C \times H \times W}$ to capture the global spatial context in the frequency domain.
# 
# $$F_{s} = DFT_{2D}(f)$$
# $$\overline{F}_{s} = F_{s} \odot \mathcal{K}$$
# $$f_{s} = IDFT_{2D}(\overline{F}_{s})$$
# 
# * $DFT_{2D}$ and $IDFT_{2D}$ denote the 2D spatial Discrete Fourier Transform and its inverse
# * $\mathcal{K} \in \mathbb{C}^{C \times H \times W}$ is the learnable spatial global filter
# * $\odot$ denotes element-wise multiplication

# In[5]:


class SpatialFISTA(nn.Module):
    """Filters features spatially in the frequency domain using a learned
    complex filter."""

    def __init__(self, channels, height, width):
        """
        Args:
            channels (int): Number of channels.
            height (int): Input feature map height.
            width (int): Input feature map width.
        """
        super().__init__()
        self.weight_real = nn.Parameter(
            torch.randn(1, channels, height, width // 2 + 1) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(1, channels, height, width // 2 + 1) * 0.02
        )

    @torch.compiler.disable
    def forward(self, f):
        """
        Args:
            f (torch.Tensor): Input features, shape (B, C, H, W).

        Returns:
            torch.Tensor: Spatially filtered features, same shape as `f`.
        """
        with torch.amp.autocast("cuda", enabled=False):  # <-- Forces out of FP16
            f = f.float()
            F_s = torch.fft.rfft2(f, dim=(-2, -1), norm="ortho")
            K = torch.complex(self.weight_real.float(), self.weight_imag.float())
            F_s_bar = F_s * K
            f_s = torch.fft.irfft2(
                F_s_bar, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1), norm="ortho"
            )
            return f_s


# <img src="./diagrams/temporalfista_architecture.svg" alt="temporalfista_architecture" style="max-width: 50%; max-height: 400px">
# 
# 
# TemporalFISTA takes the spatially filtered output $f_s \in \mathbb{R}^{T \times C \times H \times W}$ and processes it across the temporal dimension to capture motion dynamics, yielding the motion context $M$.
# 
# $$F_{t} = DFT_{1D}(f_{s})$$
# $$\overline{F}_{t} = F_{t} \odot \mathcal{K}_{t}$$
# $$\hat{f} = IDFT_{1D}(\overline{F}_{t})$$
# $$M = f \odot ||\hat{f}||_{2}$$
# 
# * $DFT_{1D}$ and $IDFT_{1D}$ denote the 1D temporal Discrete Fourier Transform and its inverse.
# * $\mathcal{K}_t \in \mathbb{C}^{T \times C \times 1 \times 1}$ is the learnable temporal global filter.
# * $||\hat{f}||_{2}$ is the $L_2$ normalization of $\hat{f}$ across the temporal dimension, representing the amplitude of temporal dynamics.

# In[6]:


class TemporalFISTA(nn.Module):
    """Filters features in the temporal dimension in the frequency domain and
    uses the result to reweight the original frames, based on fourier transform -> inverse fourier transform.
    """

    def __init__(self, frames, channels):
        """
        Args:
            frames (int): Number of frames per clip (FFT dimension).
            channels (int): Number of channels.
        """
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(1, frames, channels, 1, 1) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(1, frames, channels, 1, 1) * 0.02)

    @torch.compiler.disable
    def forward(self, f_s, f_orig):
        """
        Args:
            f_s (torch.Tensor): Features to filter temporally, shape (B, T, C, H, W).
            f_orig (torch.Tensor): Original features to reweight, shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Reweighted features, same shape as `f_orig`.
        """
        with torch.amp.autocast("cuda", enabled=False):
            f_s = f_s.float()
            f_orig = f_orig.float()
            F_t = torch.fft.fft(f_s, dim=1, norm="ortho")
            K_t = torch.complex(self.weight_real.float(), self.weight_imag.float())
            F_t_bar = F_t * K_t
            f_hat_complex = torch.fft.ifft(F_t_bar, dim=1, norm="ortho")
            f_hat = f_hat_complex.real
            f_hat_norm = torch.linalg.vector_norm(f_hat, ord=2, dim=1, keepdim=True)
            M = f_orig * f_hat_norm
            return M


# MotionGuidedSpatialConv uses the motion context $M$ to dynamically calibrate the spatial convolution weights for each frame, applying them to the original input $f$.
# 
# $$\alpha_{t} = FC(GAP_{s}(M))$$
# $$W_{t} = \alpha_{t} \cdot W_{b}$$
# $$f_{out} = W_{t} * f$$
# 
# * $GAP_{s}$ is the spatial global average pooling operation.
# * $FC$ is a fully connected layer operating across the temporal dimension for temporal modeling.
# * $\alpha_t \in \mathbb{R}^{T \times C \times 1 \times 1}$ is the derived calibration weight.
# * $W_{b} \in \mathbb{R}^{C \times C \times k^{2}}$ represents the base weight for each frame.
# * $*$ signifies the convolution operation.

# In[7]:


import math


class MotionGuidedSpatialConv(nn.Module):
    """Applies a per-frame spatial convolution whose kernel is scaled by
    motion-derived attention weights."""

    def __init__(self, channels, frames, ksize=3):
        """
        Args:
            channels (int): Number of input/output channels.
            frames (int): Number of frames per clip.
            ksize (int): Spatial kernel size. Defaults to 3.
        """
        super().__init__()
        self.channels = channels
        self.frames = frames
        self.ksize = ksize

        self.Wb = nn.Parameter(torch.Tensor(channels, channels, ksize, ksize))
        nn.init.kaiming_uniform_(self.Wb, a=math.sqrt(5))
        self.fc = nn.Linear(frames, frames)

    def forward(self, f, M):
        """
        Args:
            f (torch.Tensor): Input features, shape (B, T, C, H, W).
            M (torch.Tensor): Motion map, shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Output features, shape (B, T, C, H, W).
        """
        B, T, C, H, W = f.shape
        gap = M.mean(dim=(-2, -1))
        gap_transposed = gap.transpose(1, 2)
        alpha_t = self.fc(gap_transposed)
        alpha_t = alpha_t.transpose(1, 2).contiguous()
        alpha_t = alpha_t.view(B * T, C, 1, 1, 1)

        Wb_reshaped = self.Wb.view(1, C, C, self.ksize, self.ksize)
        Wt = alpha_t * Wb_reshaped
        Wt = Wt.view(B * T * C, C, self.ksize, self.ksize)

        f_reshaped = f.view(1, B * T * C, H, W)
        f_out = F.conv2d(f_reshaped, Wt, groups=B * T, padding=self.ksize // 2)
        f_out = f_out.view(B, T, C, H, W)

        return f_out


# FISTAblock sequentially connects the blocks to map the input $f$ to output $f_{out}$.
# 
# $$f_{s} = \text{SpatialFISTA}(f)$$
# $$M = \text{TemporalFISTA}(f_{s}, f)$$
# $$f_{out} = \text{MotionGuidedSpatialConv}(f, M)$$

# <img src="./diagrams/fista_block_architecture.png" alt="fista_block" style="max-width: 50%;">
# 
# NOTE: Based on model training, If we have these blocks, without any residual connection, the features collapse, and learning stagnates. so similar to CSPLayer's Bottleneck, I've added a residual connection to end, to prevent feature collapse.
# 

# In[8]:


class FISTABlock(nn.Module):
    """FISTA Block, based on MOCID's diagram and an additional residual connection to prevent feature collapse"""

    def __init__(self, channels, frames, height, width, ksize=3):
        super().__init__()
        self.spatial_fista = SpatialFISTA(channels, height, width)
        self.temporal_fista = TemporalFISTA(frames, channels)
        self.dynamic_conv = MotionGuidedSpatialConv(channels, frames, ksize)

    def forward(self, f):
        f_s = self.spatial_fista(f)
        M = self.temporal_fista(f_s, f)
        f_out = self.dynamic_conv(f, M)
        return f + f_out  # skip connection added (not mentioned in MOCID paper)


# <img src="./diagrams/fista_layer_architecture.png" alt="fista_layer" style="max-width: 50%;">

# In[9]:


class ConvBlock(nn.Module):
    """A depthwise 3x3 conv followed by two parallel 1x1 conv branches,
    summed together."""

    def __init__(self, channels):
        """
        Args:
            channels (int): Number of input/output channels.
        """
        super().__init__()
        self.conv_3x3 = BaseConv(channels, channels, ksize=3, stride=1, groups=channels)
        self.conv_1x1_a = BaseConv(channels, channels, ksize=1, stride=1)
        self.conv_1x1_b = BaseConv(channels, channels, ksize=1, stride=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features, shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Output features, shape (B, T, C, H, W).
        """
        B, T, C, H, W = x.shape
        x_2d = x.view(B * T, C, H, W)
        out_3x3 = self.conv_3x3(x_2d)
        branch_a = self.conv_1x1_a(out_3x3)
        branch_b = self.conv_1x1_b(out_3x3)
        out_2d = branch_a + branch_b
        return out_2d.view(B, T, C, H, W)


class FISTALayer(nn.Module):
    """Projects features to a reduced channel dimension, applies a stack of
    ConvBlock + FISTABlock pairs defined by number of block repetitions, then projects back to the original channels.
    """

    def __init__(self, channels, frames, height, width, n_blocks):
        """
        Args:
            channels (int): Number of input/output channels.
            frames (int): Number of frames per clip.
            height (int): Input feature map height.
            width (int): Input feature map width.
            n_blocks (int): Number of ConvBlock + FISTABlock pairs to stack.
        """
        super().__init__()
        hidden_channels = channels // 2

        self.proj_in = BaseConv(channels, hidden_channels, ksize=1, stride=1)
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv_model": ConvBlock(hidden_channels),
                        "fista_block": FISTABlock(
                            hidden_channels, frames, height, width
                        ),
                    }
                )
            )

        self.proj_out = BaseConv(hidden_channels, channels, ksize=1, stride=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features, shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Output features, shape (B, T, C, H, W).
        """
        B, T, C, H, W = x.shape
        x_2d = x.view(B * T, C, H, W)
        x_2d = self.proj_in(x_2d)
        x = x_2d.view(B, T, -1, H, W)

        for block_pair in self.blocks:
            x = block_pair["conv_model"](x)
            x = block_pair["fista_block"](x)

        x_2d = x.view(B * T, -1, H, W)
        x_2d = self.proj_out(x_2d)
        return x_2d.view(B, T, C, H, W)


# 
# 
# | Backbone Stage | Output Downsampling Scale | Parameter $n$ (Repetitions) | Internal Layer Sequence |
# | :--- | :--- | :---: | :--- |
# | **Layer 1** (Retained Base) | $\times \frac{1}{4}$ | $1$ | 1 BaseConv $\rightarrow$ 1 CSPLayer Block |
# | **Layer 2** (Retained Base) | $\times \frac{1}{8}$ | $2$ | 1 BaseConv $\rightarrow$ 2 CSPLayer Blocks |
# | **FISTA Layer 1** (Replaced) | $\times \frac{1}{8}$ to $\times \frac{1}{16}$ | **$2$** | Sequence of 2 [Spatial Conv Block $\rightarrow$ FISTA Block] pairs |
# | **FISTA Layer 2** (Replaced) | $\times \frac{1}{16}$ to $\times \frac{1}{32}$ | **$2$** | Sequence of 2 [Spatial Conv Block $\rightarrow$ FISTA Block] pairs |
# | **FISTA Layer 3** (Replaced) | $\times \frac{1}{32}$ | **$1$** | Sequence of 1 [Spatial Conv Block $\rightarrow$ FISTA Block] pairs |
# 
# __The SpatioTemporal backbone uses CSPDarknet 21 with an extra FISTA layer__
# 
# NOTE: original backbone, for CSPDarknet 21 [based on YoloX's source](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/models/darknet.py#L12), uses [1, 2, 2, 1] __bottleneck__ repetitions for layers [Dark 2, Dark3, Dark4, Dark5]
# 
# in SpatioTemporalBackbone, I retained, [Dark2, Dark3] but replaced the remaining with [FISTA 1, FISTA 2, FISTA 3]

# In[10]:


class SpatioTemporalBackbone(nn.Module):
    """
    Produces a clip-level feature pyramid PER TARGET FRAME (I2..I5).
    Returns (Ft, Fr_list):
        Ft       = [F5]           target feature; 3 maps [s8, s16, s32], each (B, C_k, H_k, W_k)
        Fr_list  = [F2, F3, F4]   reference features; each 3 maps [s8, s16, s32], each (B, C_k, H_k, W_k)
    """

    def __init__(self, in_channels=3, base_channels=16, frames=5, img_size=512):
        """
        Args:
            in_channels (int): Number of input image channels. Defaults to 3.
            base_channels (int): Base channel width scaled up at each stage. Defaults to 16.
            frames (int): Number of frames per clip. Defaults to 5.
            img_size (int): Input image height/width (assumed square). Defaults to 512.
        """
        super().__init__()
        self.frames = frames

        self.stem = BaseConv(in_channels, base_channels, ksize=3, stride=1)

        self.spatial_layer1 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, ksize=3, stride=2),
            CSPLayer(base_channels * 2, base_channels * 2, num_bottlenecks=1),
        )
        self.spatial_layer2 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, ksize=3, stride=2),
            CSPLayer(base_channels * 4, base_channels * 4, num_bottlenecks=2),
        )

        self.downsample1 = BaseConv(
            base_channels * 4, base_channels * 8, ksize=3, stride=2
        )
        self.fista_layer1 = FISTALayer(
            channels=base_channels * 8,
            frames=frames,
            height=img_size // 8,
            width=img_size // 8,
            n_blocks=2,
        )
        self.downsample2 = BaseConv(
            base_channels * 8, base_channels * 16, ksize=3, stride=2
        )
        self.fista_layer2 = FISTALayer(
            channels=base_channels * 16,
            frames=frames,
            height=img_size // 16,
            width=img_size // 16,
            n_blocks=2,
        )
        self.downsample3 = BaseConv(
            base_channels * 16, base_channels * 32, ksize=3, stride=2
        )
        self.fista_layer3 = FISTALayer(
            channels=base_channels * 32,
            frames=frames,
            height=img_size // 32,
            width=img_size // 32,
            n_blocks=1,
        )

    def _encode_spatial(self, frame):
        """Encodes a single frame through the shared spatial stack (stem +
        two spatial layers) to produce a stride-4 feature map.

        Args:
            frame (torch.Tensor): Single frame, shape (B, C, H, W).

        Returns:
            torch.Tensor: Spatial feature at stride 4, shape (B, Cf, Hf, Wf).
        """
        # frame: (B, C, H, W) -> spatial feature at stride 4
        s = self.stem(frame)
        s = self.spatial_layer1(s)
        s = self.spatial_layer2(s)
        return s

    def _downsample_clip(self, conv, clip):
        """Applies a 2D downsampling conv independently to every frame of a clip.

        Args:
            conv (nn.Module): 2D conv module to apply per frame.
            clip (torch.Tensor): Clip features, shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Downsampled clip features, shape (B, T, Co, Ho, Wo).
        """
        # clip: (B, T, C, H, W) -> per-frame 2D downsample
        B, T, C, H, W = clip.shape
        out = conv(clip.reshape(B * T, C, H, W))
        _, Co, Ho, Wo = out.shape
        return out.view(B, T, Co, Ho, Wo)

    def _run_fista(self, clip):
        """Runs the clip through three successive downsample + FISTALayer
        stages to produce a multi-scale feature pyramid.

        Args:
            clip (torch.Tensor): Clip features, shape (B, 5, Cf, Hf, Wf).

        Returns:
            tuple: (o1, o2, o3), three scale outputs, each of shape
            (B, 5, C_k, H_k, W_k).
        """
        # clip: (B, 5, Cf, Hf, Wf) -> three scale outputs, each (B, 5, C_k, H_k, W_k)
        c = self._downsample_clip(self.downsample1, clip)
        o1 = self.fista_layer1(c)
        c = self._downsample_clip(self.downsample2, o1)
        o2 = self.fista_layer2(c)
        c = self._downsample_clip(self.downsample3, o2)
        o3 = self.fista_layer3(c)
        return o1, o2, o3

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input clip, shape (B, T, C, H, W) with T == 5,
                ordered [I1, I2, I3, I4, I5] where I5 is the target frame.

        Returns:
            tuple: (Ft, Fr_list)
                Ft (list[torch.Tensor]): Target-frame (I5) features at 3
                    scales, each shape (B, C_k, H_k, W_k).
                Fr_list (list[torch.Tensor]): Reference-frame (I1..I4)
                    features at 3 scales, each shape (B, 4, C_k, H_k, W_k).
        """
        B, T, C, H, W = x.shape  # T == 5

        # encode each frame through the shared spatial stack
        s = [self._encode_spatial(x[:, i]) for i in range(T)]

        # single natural clip: [I1, I2, I3, I4, I5], target last
        clip = torch.stack(s, dim=1)  # (B, 5, Cf, Hf, Wf)

        o1, o2, o3 = self._run_fista(clip)  # each (B, 5, C_k, H_k, W_k)

        # target = last frame (I5); references = first four (I1..I4)
        Ft = [o1[:, -1], o2[:, -1], o3[:, -1]]  # 3 maps, each (B, C, H, W)
        Fr_list = [o1[:, :-1], o2[:, :-1], o3[:, :-1]]  # 3 maps, each (B, 4, C, H, W)

        return Ft, Fr_list


# YoloXHead for Detection
# ```tex
# Ge, Zheng & Liu, Songtao & Wang, Feng & Li, Zeming & Sun, Jian. (2021). YOLOX: Exceeding YOLO Series in 2021. 10.48550/arXiv.2107.08430. 
# ```
# IOULoss, YoloLoss and YoloPAFPN taken directly from [SSTnet](https://github.com/UESTC-nnLab/SSTNet) which has taken it from [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)

# In[11]:


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[16, 32, 64], act="silu"):
        super().__init__()
        Conv = BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


# In[ ]:


import torch
import torch.nn as nn


class NWDLoss(nn.Module):
    """
    Normalized Gaussian Wasserstein Distance (NWD) Loss.
    Designed specifically for tiny object detection.
    Models bounding boxes as 2D Gaussian distributions and measures the distance.
    Reference: https://arxiv.org/pdf/2110.13389v2.pdf
    """

    def __init__(self, reduction="none", C=12.8):
        """
        Args:
            reduction: "none", "mean", or "sum"
            C: Dataset-specific constant (average absolute size of the dataset).
               The paper uses 12.8 for the AI-TOD dataset.
        """
        super(NWDLoss, self).__init__()
        self.reduction = reduction
        self.C = C

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted bounding boxes [N, 4] format (cx, cy, w, h)
            target: Ground truth bounding boxes [N, 4] format (cx, cy, w, h)
        """
        assert (
            pred.shape[0] == target.shape[0]
        ), "Predictions and targets must have same batch size"

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # Extract (cx, cy, w, h)
        cx_p, cy_p, w_p, h_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        cx_t, cy_t, w_t, h_t = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Calculate the 2nd order Wasserstein distance squared (W_2^2)
        # Equation (7) from the paper
        w2_sq = (
            (cx_p - cx_t) ** 2
            + (cy_p - cy_t) ** 2
            + ((w_p - w_t) / 2) ** 2
            + ((h_p - h_t) / 2) ** 2
        )

        # Normalize into a similarity metric (0 to 1)
        # Equation (8) from the paper
        nwd = torch.exp(
            -torch.sqrt(w2_sq + 1e-10) / self.C
        )  # added 1e-10 for numerical stability

        # Equation (9) from the paper: Final regression loss
        loss = 1 - nwd

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming NWDLoss is imported from the other file
# from nwd_loss import NWDLoss


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, fp16, strides=[8, 16, 32], nwd_constant=12.8):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

        # --- REPLACED IOUloss WITH NWDLoss ---
        self.nwd_loss = NWDLoss(reduction="none", C=nwd_constant)
        self.nwd_constant = nwd_constant
        # -------------------------------------

        self.grids = [torch.zeros(1)] * len(strides)
        self.fp16 = fp16

    @torch.compiler.disable
    def forward(self, inputs, labels=None):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(
            x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1)
        )

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(
                [torch.arange(hsize), torch.arange(wsize)], indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)

        xy = (output[..., :2] + grid.type_as(output)) * stride
        wh = torch.exp(torch.clamp(output[..., 2:4], max=20.0)) * stride
        rest = output[..., 4:]

        output = torch.cat([xy, wh, rest], dim=-1)
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_metrics_this_matching,  # Replaced IOU variable naming
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    cls_preds_per_image,
                    obj_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ).float() * pred_metrics_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)

        # --- CALCULATE NWD REGRESSION LOSS INSTEAD OF IOU ---
        loss_box = (self.nwd_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        # ----------------------------------------------------

        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum()

        reg_weight = 5.0
        # Replaced loss_iou with loss_box
        loss = reg_weight * loss_box + loss_obj + loss_cls

        return loss / num_fg

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if num_in_boxes_anchor == 0:
            return (
                gt_classes.new_zeros((0,), dtype=torch.long),
                gt_bboxes_per_image.new_zeros((total_num_anchors,), dtype=torch.bool),
                gt_bboxes_per_image.new_zeros((0,), dtype=torch.float),
                gt_classes.new_zeros((0,), dtype=torch.long),
                0,
            )

        # --- NWD CALCULATION FOR LABEL ASSIGNMENT ---
        # Instead of self.bboxes_iou, we calculate pairwise NWD
        # between GTs and predicted boxes to build the cost matrix
        pair_wise_nwd = self.bboxes_nwd(gt_bboxes_per_image, bboxes_preds_per_image)

        # We invert the NWD to get a loss (higher NWD = lower loss)
        # NWD is 0-1 bounded, similar to IoU.
        pair_wise_nwd_loss = -torch.log(pair_wise_nwd + 1e-8)
        # --------------------------------------------

        if self.fp16:
            with torch.cuda.amp.autocast(enabled=False):
                cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
                )
                gt_cls_per_image = (
                    F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                    .float()
                    .unsqueeze(1)
                    .repeat(1, num_in_boxes_anchor, 1)
                )
                pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
                ).sum(-1)
        else:
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
                * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
            )
            gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

        # Cost now uses the NWD loss instead of IOU loss
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_nwd_loss
            + 100000.0 * (~is_in_boxes_and_center).float()
        )

        num_fg, gt_matched_classes, pred_metrics_this_matching, matched_gt_inds = (
            # Pass pair_wise_nwd so dynamic_k_matching can use it for Top-K
            self.dynamic_k_matching(cost, pair_wise_nwd, gt_classes, num_gt, fg_mask)
        )

        del pair_wise_cls_loss, cost, pair_wise_nwd, pair_wise_nwd_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_metrics_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def bboxes_nwd(self, bboxes_a, bboxes_b):
        """
        Calculates pair-wise NWD instead of pair-wise IOU.
        bboxes_a (GT): [num_gt, 4] format (cx, cy, w, h)
        bboxes_b (Preds): [num_anchors, 4] format (cx, cy, w, h)
        """
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        # Reshape to calculate combinations
        # bboxes_a: [num_gt, 1, 4]
        # bboxes_b: [1, num_anchors, 4]
        cx_a = bboxes_a[:, None, 0]
        cy_a = bboxes_a[:, None, 1]
        w_a = bboxes_a[:, None, 2]
        h_a = bboxes_a[:, None, 3]

        cx_b = bboxes_b[:, 0]
        cy_b = bboxes_b[:, 1]
        w_b = bboxes_b[:, 2]
        h_b = bboxes_b[:, 3]

        # W_2^2 distance calculation
        w2_sq = (
            (cx_a - cx_b) ** 2
            + (cy_a - cy_b) ** 2
            + ((w_a - w_b) / 2) ** 2
            + ((h_a - h_b) / 2) ** 2
        )

        # NWD metric (bounded 0 to 1)
        pair_wise_nwd = torch.exp(-torch.sqrt(w2_sq + 1e-10) / self.nwd_constant)

        return pair_wise_nwd

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        center_radius=2.5,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = (
            ((x_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            ((y_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_metrics, gt_classes, num_gt, fg_mask):
        """
        pair_wise_metrics: Previously pair_wise_ious. Now representing pair_wise_nwd.
        """
        matching_matrix = torch.zeros_like(cost)

        n_candidate_k = min(10, pair_wise_metrics.size(1))
        # Top-k matches are now based on NWD instead of IoU
        topk_metrics, _ = torch.topk(pair_wise_metrics, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(
            topk_metrics.sum(1).int(), min=1, max=pair_wise_metrics.size(1)
        )

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_metrics, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_metrics_this_matching = (matching_matrix * pair_wise_metrics).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_metrics_this_matching, matched_gt_inds


# Lin et al 2027 FPN, with connections as listed in diagram for FPN.
# 
# Comes from average pooled (spatio-temporal pooling) features from backbone.

# In[14]:


class FPN(nn.Module):
    """top-down FPN (Lin et al. 2017): lateral 1x1 -> common dim,
    top-down add, 3x3 smoothing. Output convs project back to per-level channels
    [c3,c4,c5] so the existing YOLOXHead stays unchanged.

    forward(xT, ff): ff = temporally-pooled F_f, added into the pyramid per scale.
        xT[k], ff[k] : (B, c_k, H_k, W_k)
        returns [P3, P4, P5] with channels [c3, c4, c5]
    """

    def __init__(self, ch, fpn_dim=256):
        super().__init__()
        c3, c4, c5 = ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # lateral 1x1: c_i -> fpn_dim
        self.l3 = BaseConv(c3, fpn_dim, 1, 1)
        self.l4 = BaseConv(c4, fpn_dim, 1, 1)
        self.l5 = BaseConv(c5, fpn_dim, 1, 1)
        # 3x3 smoothing + project back to per-level channels for the head
        self.o3 = BaseConv(fpn_dim, c3, 3, 1)
        self.o4 = BaseConv(fpn_dim, c4, 3, 1)
        self.o5 = BaseConv(fpn_dim, c5, 3, 1)

    def forward(self, xT, ff=None):
        x3, x4, x5 = xT
        if ff is not None:  # F_f injection (channels match c_i)
            x3 = x3 + ff[0]
            x4 = x4 + ff[1]
            x5 = x5 + ff[2]

        lat3, lat4, lat5 = self.l3(x3), self.l4(x4), self.l5(x5)  # -> fpn_dim
        m5 = lat5  # top-down pathway
        m4 = lat4 + self.up(m5)
        m3 = lat3 + self.up(m4)
        return [self.o3(m3), self.o4(m4), self.o5(m5)]  # -> [c3,c4,c5]


# <img src="./diagrams/mocid_architecture.png" alt="mocid" style="max-height: 200px;">
# 

# In[15]:


class TemporalPooling(nn.Module):
    """Simple 3D-conv temporal pooling: collapse T -> 1 per scale with a
    depthwise Conv3d whose temporal kernel spans the whole clip.

    Input  feats_by_scale[k] : (B, T, C_k, H, W)
    Output list of           : (B, C_k, H, W)
    """

    def __init__(self, channels, frames=5, ksize=3):
        super().__init__()
        pad = ksize // 2
        # depthwise: temporal kernel = T (valid -> collapses T), spatial = ksize
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(
                    c,
                    c,
                    (frames, ksize, ksize),
                    padding=(0, pad, pad),
                    groups=c,
                    bias=False,
                )
                for c in channels
            ]
        )

    def forward(self, feats_by_scale):
        pooled = []
        for conv, feat in zip(self.convs, feats_by_scale):  # feat (B,T,C,H,W)
            x = feat.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
            pooled.append(conv(x).squeeze(2))  # (B,C,H,W)
        return pooled


from dam import DisplacementNet


class MOCID(nn.Module):
    def __init__(
        self, num_classes=1, num_frames=5, img_size=512, base_channels=16, d_state=16
    ):
        super().__init__()
        ch = [base_channels * 8, base_channels * 16, base_channels * 32]
        self.backbone = SpatioTemporalBackbone(3, base_channels, num_frames, img_size)
        self.pool = TemporalPooling(ch, frames=num_frames)
        self.disp = DisplacementNet(ch, d_state=d_state, expand=1, theta=0.7)
        self.fpn = FPN(ch)
        self.head = YOLOXHead(num_classes, width=0.5, in_channels=[c * 2 for c in ch])
        self.loss_fn = YOLOLoss(num_classes, fp16=False, strides=[8, 16, 32])

    def forward(self, clip, labels=None, use_dam=True):
        Ft, Fr_list = self.backbone(clip)

        # rebuild per-scale (B, T, C, H, W) with target as the last frame
        feats_by_scale = [
            torch.cat([Fr_list[k], Ft[k].unsqueeze(1)], dim=1)  # (B, 4+1, C, H, W)
            for k in range(3)
        ]

        F_T = Ft  # 3 target maps, each (B, C, H, W)

        if use_dam:
            with torch.autocast("cuda", enabled=False):  # scan is fp16-unstable
                motion_features = self.disp([f.float() for f in feats_by_scale])
        else:
            motion_features = feats_by_scale

        F_f = self.pool(motion_features)
        for t in F_f:
            assert torch.isfinite(t).all(), "non-finite in F_f (DAM output)"
        outs = self.head(self.fpn(F_T, F_f))

        if labels is not None:
            with torch.amp.autocast("cuda", enabled=False):
                return self.loss_fn([o.float() for o in outs], labels)
        return outs


# In[16]:


m = MOCID()


def pm(model, dam):
    it = (
        model.parameters()
        if dam
        else (p for n, p in model.named_parameters() if not n.startswith("disp."))
    )
    return sum(p.numel() for p in it) / 1e6


print(f".+FISTA (no DAM) : {pm(m, False):.2f} M   (target 9.45)")
print(f".+FISTA+DAM      : {pm(m, True):.2f} M    (target 13.05)")


# In[17]:


import csv
import time
import os
import argparse
import sys
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from eval import evaluate, collate_eval


def _strip_compile(sd):
    """Removes the '_orig_mod.' prefix that torch.compile adds to state_dict keys.

    Args:
        sd (dict): Model state_dict, possibly with compiled key prefixes.

    Returns:
        dict: State_dict with prefixes stripped.
    """
    return {
        k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
        for k, v in sd.items()
    }


def save_ckpt(path, model, opt, sched, ep, stage, tag, ap50=None):
    """Saves a training checkpoint atomically (write to .tmp then rename).

    Args:
        path (str): Destination checkpoint path.
        model (nn.Module): Model whose state_dict is saved.
        opt (torch.optim.Optimizer): Optimizer whose state_dict is saved.
        sched (torch.optim.lr_scheduler._LRScheduler): LR scheduler to save.
        ep (int): Current epoch index.
        stage (int): Current training stage.
        tag (str): Run tag/name.
        ap50 (float, optional): AP50 metric to store with the checkpoint. Defaults to None.

    Returns:
        None
    """
    torch.save(
        {
            "model": _strip_compile(model.state_dict()),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": ep,
            "stage": stage,
            "tag": tag,
            "ap50": ap50,
            "rng_cpu": torch.get_rng_state().cpu(),
            "rng_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()],
        },
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def _restore_rng(ckpt):
    """Restores CPU/CUDA RNG state from a checkpoint, if present.

    Args:
        ckpt (dict): Loaded checkpoint dictionary.

    Returns:
        None
    """
    try:
        if "rng_cpu" in ckpt:
            torch.set_rng_state(ckpt["rng_cpu"].cpu().to(torch.uint8))
        if "rng_cuda" in ckpt:
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in ckpt["rng_cuda"]]
            )
    except Exception as e:
        print(f"[warn] skipping RNG restore ({e})")


def log_eval(csv_path, row):
    """Appends an evaluation result row to a CSV file, writing a header if new.

    Args:
        csv_path (str): Path to the CSV log file.
        row (dict): Row of values to log; keys are used as CSV columns.

    Returns:
        None
    """
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# In[18]:


import math


def _sgd(model, cfg, epochs):
    """Builds an SGD+Nesterov optimizer with a linear warmup + cosine decay
    learning rate schedule.

    Args:
        model (nn.Module): Model whose trainable parameters are optimized.
        cfg: Config object with LR_INIT, MOMENTUM, WEIGHT_DECAY, WARMUP_EPOCHS, MIN_LR.
        epochs (int): Total number of training epochs (schedule length).

    Returns:
        tuple: (opt, sched)
            opt (torch.optim.SGD): The optimizer.
            sched (torch.optim.lr_scheduler.LambdaLR): The LR scheduler.
    """
    # Filter out frozen parameters dynamically based on the stage
    params = [p for p in model.parameters() if p.requires_grad]

    # Mirror SSTNet: SGD + Nesterov
    opt = torch.optim.SGD(
        params,
        lr=cfg.LR_INIT,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        nesterov=True,
    )

    warmup = cfg.WARMUP_EPOCHS
    total_steps = epochs

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup

        # Cosine annealing from LR_INIT down to MIN_LR
        progress = (ep - warmup) / (total_steps - warmup)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        # lambda returns a multiplier for LR_INIT, not the raw LR
        min_lr_ratio = cfg.MIN_LR / cfg.LR_INIT
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


# In[ ]:


from torch.utils.data import Subset


def set_stage(model, stage):
    """stage 1 = FISTA : train backbone + pool + fpn + head, freeze disp (DAM)
    stage 2 = DAM   : freeze backbone (FISTA), train disp + pool + fpn + head"""
    for p in model.parameters():
        p.requires_grad_(False)
    if stage == 1:
        train = [model.backbone, model.pool, model.fpn, model.head]
    elif stage == 2:
        train = [
            m
            for m in [getattr(model, "disp", None), model.pool, model.fpn, model.head]
            if m is not None
        ]
    else:
        raise ValueError(stage)
    for m in train:
        for p in m.parameters():
            p.requires_grad_(True)


def _run(
    model,
    loader,
    val_loader,
    opt,
    sched,
    epochs,
    device,
    use_dam,
    tag,
    stage,
    out_dir=".",
    start_ep=0,
    ckpt_every=5,
    ckpt_path="ckpt_last.pth",
    final_path=None,
    best_path=None,
    eval_every=20,
    csv_path="eval_log.csv",
    do_eval=True,
):
    """One training stage. All checkpoints + the eval CSV are written into out_dir.
    do_eval=False skips validation (used for smoke tests)."""
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, ckpt_path)
    csv_path = os.path.join(out_dir, csv_path)
    if final_path is not None:
        final_path = os.path.join(out_dir, final_path)
    if best_path is not None:
        best_path = os.path.join(out_dir, best_path)

    scaler = torch.amp.GradScaler("cuda")

    best_ap50 = -1.0
    if best_path is not None and os.path.exists(best_path):
        try:
            prev = torch.load(best_path, map_location="cpu")
            best_ap50 = prev.get("ap50", -1.0) or -1.0
            print(f"[{tag}] existing best AP50 = {best_ap50:.2f}")
        except Exception as e:
            print(f"[warn] could not read best ckpt ({e})")

    for ep in range(start_ep, epochs):
        model.train()
        if use_dam:
            model.backbone.eval()  # frozen FISTA: no BN stat updates

        tot = nb = 0
        pbar = tqdm(
            loader, desc=f"[{tag}] {ep+1}/{epochs}", leave=False, mininterval=15
        )
        for clip, labels in pbar:
            clip = clip.to(device, non_blocking=True)
            labels = [l.to(device, non_blocking=True) for l in labels]

            with torch.amp.autocast("cuda"):
                loss = model(clip, labels, use_dam=use_dam)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=10.0
            )
            scaler.step(opt)
            scaler.update()

            tot += loss.item()
            nb += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{tot/nb:.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        sched.step()
        avg = tot / max(nb, 1)
        print(
            f"[{tag}] {ep+1}/{epochs}  loss {avg:.4f}  lr {opt.param_groups[0]['lr']:.2e}"
        )

        if (ep + 1) % ckpt_every == 0 or (ep + 1) == epochs:
            save_ckpt(ckpt_path, model, opt, sched, ep, stage, tag)

        if do_eval and ((ep + 1) % eval_every == 0 or (ep + 1) == epochs):
            ap50, f1 = evaluate(
                model, val_loader, device, use_dam, strides=[8, 16, 32], num_classes=1
            )
            log_eval(
                csv_path,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": stage,
                    "tag": tag,
                    "epoch": ep + 1,
                    "use_dam": use_dam,
                    "ap50": round(ap50 * 100, 2),
                    "f1": round(f1 * 100, 2),
                    "avg_loss": round(avg, 4),
                },
            )
            print(f"[{tag}] eval @ ep{ep+1}: AP50 {ap50*100:.2f}  F1 {f1*100:.2f}")
            if best_path is not None and ap50 * 100 > best_ap50:
                best_ap50 = ap50 * 100
                save_ckpt(best_path, model, opt, sched, ep, stage, tag, ap50=best_ap50)
                print(f"[{tag}] *** new best AP50 {best_ap50:.2f} -> {best_path}")

    if final_path is not None:
        save_ckpt(final_path, model, opt, sched, epochs - 1, stage, tag)
        print(f"[{tag}] saved stage checkpoint -> {final_path}")


def train_mocid(cfg, tag="default", train_loader=None, val_loader=None, do_eval=True):
    """Two-stage training: (1) FISTA-only, (2) freeze FISTA -> train DAM.
    All artifacts go into runs/<tag>/. Pass loaders to override DAUB (smoke tests)."""
    model = MOCID(num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)
    model = torch.compile(model)  # <-- restored
    out_dir = os.path.join("runs", tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] artifacts -> {out_dir}/")

    if train_loader is None:
        train_ds = MOCIDDataset(
            cfg.train_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=True
        )
        val_ds = MOCIDDataset(
            cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_mocid,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_eval,
        )

    # ---------- Stage 1: FISTA ----------
    print("=" * 25, "STAGE 1: FISTA", "=" * 25)
    set_stage(model, 1)
    opt, sched = _sgd(model, cfg, cfg.EPOCHS_SPTBACKBONE)
    _run(
        model,
        train_loader,
        val_loader,
        opt,
        sched,
        cfg.EPOCHS_SPTBACKBONE,
        device,
        use_dam=False,
        tag=f"{tag}-fista",
        stage=1,
        out_dir=out_dir,
        ckpt_path="fista_last.pth",
        final_path="fista.pth",
        best_path="fista_best.pth",
        csv_path="eval_log.csv",
        eval_every=cfg.EVAL_EVERY,
        do_eval=do_eval,
    )

    # ---------- Stage 2: freeze FISTA -> train DAM ----------
    print("=" * 25, "STAGE 2: DAM", "=" * 25)
    set_stage(model, 2)
    opt, sched = _sgd(model, cfg, cfg.EPOCHS_DAM)  # rebuilt AFTER freezing
    _run(
        model,
        train_loader,
        val_loader,
        opt,
        sched,
        cfg.EPOCHS_DAM,
        device,
        use_dam=True,
        tag=f"{tag}-dam",
        stage=2,
        out_dir=out_dir,
        ckpt_path="dam_last.pth",
        final_path="dam.pth",
        best_path="dam_best.pth",
        csv_path="eval_log.csv",
        eval_every=cfg.EVAL_EVERY,
        do_eval=do_eval,
    )
    return model


# In[ ]:


def _get_tag():
    if "ipykernel" in sys.modules:
        return "default"
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    args, _ = ap.parse_known_args()
    return args.tag


if __name__ == "__main__":
    model = train_mocid(Config(), tag=_get_tag())


# Displacement Net Module (Probably Wrong, and not used in training yet)
