#!/usr/bin/env python
# coding: utf-8

# In[48]:


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

# In[49]:


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
    LR_DAM = 1e-3  # stage-2 (DAM) LR; 1e-2 diverges a fresh Mamba branch
    WARMUP_EPOCHS = 6
    MOMENTUM = 0.937
    WEIGHT_DECAY = 5e-4
    EPOCHS_SPTBACKBONE = 100
    EPOCHS_DAM = 100
    EVAL_EVERY = 2
    TRACK_BEST_AFTER = 40

    train_path = "../../datasets/IRDST_mocid/train_IRDST.txt"
    val_path = "../../datasets/IRDST_mocid/val_IRDST.txt"


# 1. data augmentation: random flipping of video clips
# 2. clips batched with T = 5, with T - 1 reference clips
# 3. clip starts with 5th frame, since we want data to have spatio-temporal consistency

# In[50]:


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
        in each clip carries the target bounding box(es) and class label(s).
        (Works with both DAUB and IRDST.)

        Each annotation line is "<img_path> <box> [<box> ...]" where every box is
        "xmin,ymin,xmax,ymax,class_id". Frames with multiple boxes are kept.
        Clips are only formed from T strictly-consecutive frames, so gaps in the
        sequence (missing images) never get bridged into one clip.
        """
        self.T = T
        self.img_size = img_size
        self.is_train = is_train
        self.clips = []

        if not os.path.exists(annotations_file):
            print(f"Warning: {annotations_file} not found.")
            return

        # seq -> {frame_num -> {"path": str, "boxes": [[x,y,x,y,cls], ...]}}
        # one entry per frame_num, accumulating all boxes on that frame.
        sequences = defaultdict(dict)
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 2:
                    continue
                img_path = parts[0]
                seq_id = os.path.basename(os.path.dirname(img_path))
                frame_num = int(os.path.basename(img_path).split(".")[0])
                boxes = [list(map(int, b.split(","))) for b in parts[1:]]

                frame = sequences[seq_id].get(frame_num)
                if frame is None:
                    sequences[seq_id][frame_num] = {"path": img_path, "boxes": boxes}
                else:
                    frame["boxes"].extend(boxes)  # same frame seen again

        for seq_id, frame_map in sequences.items():
            frames = [
                {"frame_num": fn, "path": fr["path"], "boxes": fr["boxes"]}
                for fn, fr in sorted(frame_map.items())
            ]
            for i in range(self.T - 1, len(frames)):
                window = frames[i - (self.T - 1) : i + 1]
                # keep only gap-free windows: span must equal T-1
                if window[-1]["frame_num"] - window[0]["frame_num"] != self.T - 1:
                    continue
                self.clips.append(window)

        print(
            f"Loaded {len(self.clips)} clips of length {self.T} from {annotations_file}"
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """Reads T frames, resizes to img_size, optionally applies a synchronized
        horizontal flip (train), and rescales the target frame's box(es).

        Returns:
            imgs_tensor (FloatTensor): (T, 3, H, W), RGB in [0, 1].
            target (dict):
                "boxes"  (FloatTensor): (N, 4) [xmin, ymin, xmax, ymax], resized coords.
                "labels" (LongTensor):  (N,)   class ids of the target frame.
        """
        clip_data = self.clips[idx]
        target_info = clip_data[-1]

        sample_img = cv2.imread(clip_data[0]["path"], cv2.IMREAD_COLOR)
        if sample_img is None:
            raise FileNotFoundError(f"Could not read image at {clip_data[0]['path']}")
        orig_h, orig_w = sample_img.shape[:2]
        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h

        do_flip = self.is_train and (random.random() > 0.5)

        boxes, labels = [], []
        for xmin, ymin, xmax, ymax, cls in target_info["boxes"]:
            xmin = round(xmin * scale_x)
            xmax = round(xmax * scale_x)
            ymin = round(ymin * scale_y)
            ymax = round(ymax * scale_y)
            if do_flip:
                xmin, xmax = self.img_size[0] - xmax, self.img_size[0] - xmin
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls)

        imgs = []
        for frame in clip_data:
            img = cv2.imread(frame["path"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            if do_flip:
                img = cv2.flip(img, 1)
            img = img.astype(np.float32) / 255.0
            imgs.append(np.transpose(img, (2, 0, 1)))

        imgs_tensor = torch.tensor(np.array(imgs), dtype=torch.float32)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return imgs_tensor, target


def collate_mocid(batch):
    """Stacks clips and converts each target's boxes to (cx, cy, w, h, class_id).

    Returns:
        clips (FloatTensor): (B, T, 3, H, W).
        labels (list[FloatTensor]): per-sample (num_boxes, 5) = [cx, cy, w, h, cls].
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

# In[51]:


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

# In[52]:


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

# In[53]:


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

            # learnable global filter, to separate motion of targets from noise
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

# In[54]:


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

# In[55]:


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

# In[56]:


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

# In[57]:


class SpatioTemporalBackbone(nn.Module):
    """
    Clip-level feature pyramid. Scale reduction (matches the diagram):
        Spatial Layers -> 1/4   |   FISTA1 -> 1/8   |   FISTA2 -> 1/16   |   FISTA3 -> 1/32
    Each FISTA stage = a per-frame stride-2 downsample followed by a FISTALayer.
    Returns (Ft, Fr_list):
        Ft      = [F5]        target features; 3 maps [s8,s16,s32], each (B, C_k, H_k, W_k)
        Fr_list = [F2,F3,F4]  reference features; 3 maps, each (B, 4, C_k, H_k, W_k)
    """

    def __init__(self, in_channels=3, base_channels=16, frames=5, img_size=512):
        super().__init__()
        self.frames = frames

        # ---- Spatial Layers: full-res -> 1/4 ----
        self.stem = BaseConv(in_channels, base_channels, ksize=3, stride=1)  # 1/1
        self.spatial_layer1 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, ksize=3, stride=2),  # 1/2
            CSPLayer(base_channels * 2, base_channels * 2, num_bottlenecks=1),
        )
        self.spatial_layer2 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, ksize=3, stride=2),  # 1/4
            CSPLayer(base_channels * 4, base_channels * 4, num_bottlenecks=2),
        )

        # ---- FISTA stage 1: 1/4 -> 1/8  (downsample halves H,W; FISTA keeps scale) ----
        self.downsample1 = BaseConv(
            base_channels * 4, base_channels * 8, ksize=3, stride=2
        )
        self.fista_layer1 = FISTALayer(
            channels=base_channels * 8,
            frames=frames,
            height=img_size // 8,
            width=img_size // 8,
            n_blocks=4,
        )

        # ---- FISTA stage 2: 1/8 -> 1/16 ----
        self.downsample2 = BaseConv(
            base_channels * 8, base_channels * 16, ksize=3, stride=2
        )
        self.fista_layer2 = FISTALayer(
            channels=base_channels * 16,
            frames=frames,
            height=img_size // 16,
            width=img_size // 16,
            n_blocks=4,
        )

        # ---- FISTA stage 3: 1/16 -> 1/32 ----
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
        s = self.stem(frame)
        s = self.spatial_layer1(s)
        s = self.spatial_layer2(s)
        return s  # (B, base*4, H/4, W/4)

    def _fista_stage(self, down, fista, clip):
        """One stage: per-frame stride-2 downsample, then FISTALayer.
        (B, T, C, H, W) -> (B, T, C', H/2, W/2)."""
        B, T, C, H, W = clip.shape
        d = down(clip.reshape(B * T, C, H, W))
        d = d.view(B, T, *d.shape[1:])
        return fista(d)

    def forward(self, x):
        B, T, C, H, W = x.shape  # T == 5, target last (I5)

        s = [self._encode_spatial(x[:, i]) for i in range(T)]
        clip = torch.stack(s, dim=1)  # (B, 5, base*4, H/4, W/4)

        o1 = self._fista_stage(self.downsample1, self.fista_layer1, clip)  # 1/8
        o2 = self._fista_stage(self.downsample2, self.fista_layer2, o1)  # 1/16
        o3 = self._fista_stage(self.downsample3, self.fista_layer3, o2)  # 1/32

        Ft = [o1[:, -1], o2[:, -1], o3[:, -1]]  # target (I5) at 3 scales
        Fr_list = [o1[:, :-1], o2[:, :-1], o3[:, :-1]]  # references (I1..I4)
        return Ft, Fr_list


# YoloXHead for Detection
# ```tex
# Ge, Zheng & Liu, Songtao & Wang, Feng & Li, Zeming & Sun, Jian. (2021). YOLOX: Exceeding YOLO Series in 2021. 10.48550/arXiv.2107.08430. 
# ```
# IOULoss, YoloLoss and YoloPAFPN taken directly from [SSTnet](https://github.com/UESTC-nnLab/SSTNet) which has taken it from [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)

# In[58]:


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

        self.initialize_biases(1e-2)  # AFTER the loop: covers all scales

    def initialize_biases(self, prior_prob=1e-2):
        b = -math.log((1 - prior_prob) / prior_prob)  # ≈ -4.595
        for conv in self.obj_preds:
            nn.init.constant_(conv.bias, b)
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, b)

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


# In[59]:


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            b1_cxy = pred[:, :2]
            b2_cxy = target[:, :2]
            center_distance = torch.sum(torch.pow((b1_cxy - b2_cxy), 2), axis=-1)
            enclose_mins = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            enclose_maxes = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(br))
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * (center_distance) / torch.clamp(
                enclose_diagonal, min=1e-6
            )
            v = (4 / (torch.pi**2)) * torch.pow(
                (
                    torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-6))
                    - torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-6))
                ),
                2,
            )
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - alpha * v
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


# In[ ]:


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, fp16, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
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
                    pred_ious_this_matching,
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
                ).float() * pred_ious_this_matching.unsqueeze(-1)
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
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        self.last_parts = (
            float(reg_weight * loss_iou / num_fg),
            float(loss_obj / num_fg),
            float(loss_cls / num_fg),
            float(num_fg),
        )
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

        pair_wise_ious = self.bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, False
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

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
                _cls = torch.nan_to_num(cls_preds_.sqrt_(), nan=0.0).clamp_(0.0, 1.0)
                pair_wise_cls_loss = F.binary_cross_entropy(
                    _cls, gt_cls_per_image, reduction="none"
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
            _cls = torch.nan_to_num(cls_preds_.sqrt_(), nan=0.0).clamp_(0.0, 1.0)
            pair_wise_cls_loss = F.binary_cross_entropy(
                _cls, gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center).float()
        )

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = (
            self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

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

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(
            topk_ious.sum(1).int(), min=3, max=pair_wise_ious.size(1)
        )

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

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

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


# Lin et al 2017 FPN, with connections as listed in diagram for FPN.
# 
# Comes from average pooled (spatio-temporal pooling) features from backbone.

# In[61]:


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

# In[62]:


# class TemporalPooling(nn.Module):
#     """Simple 3D-conv temporal pooling: collapse T -> 1 per scale with a
#     depthwise Conv3d whose temporal kernel spans the whole clip.

#     Input  feats_by_scale[k] : (B, T, C_k, H, W)
#     Output list of           : (B, C_k, H, W)
#     """

#     def __init__(self, channels, frames=5, ksize=3):
#         super().__init__()
#         pad = ksize // 2
#         # depthwise: temporal kernel = T (valid -> collapses T), spatial = ksize
#         self.convs = nn.ModuleList(
#             [
#                 nn.Conv3d(
#                     c,
#                     c,
#                     (frames, ksize, ksize),
#                     padding=(0, pad, pad),
#                     groups=c,
#                     bias=False,
#                 )
#                 for c in channels
#             ]
#         )

#     def forward(self, feats_by_scale):
#         pooled = []
#         for conv, feat in zip(self.convs, feats_by_scale):  # feat (B,T,C,H,W)
#             x = feat.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
#             pooled.append(conv(x).squeeze(2))  # (B,C,H,W)
#         return pooled


class TemporalPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats_by_scale):
        pooled = []
        for feat in feats_by_scale:
            pooled.append(feat.amax(dim=1))
        return pooled


from dam import DisplacementNet


class MOCID(nn.Module):
    def __init__(
        self, num_classes=1, num_frames=5, img_size=512, base_channels=16, d_state=32
    ):
        super().__init__()
        ch = [base_channels * 8, base_channels * 16, base_channels * 32]
        self.backbone = SpatioTemporalBackbone(3, base_channels, num_frames, img_size)
        self.pool = TemporalPooling()
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


# In[63]:


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


# In[64]:


import math
from copy import deepcopy


class ModelEMA:
    """EMA of model params + buffers (BN running stats included). Unwraps
    torch.compile so the shadow holds clean weights. Eval / checkpoint self.ema
    instead of the live model for a smoother, less jittery metric."""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0, freeze_backbone=False):
        raw = getattr(model, "_orig_mod", model)
        self.ema = deepcopy(raw).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        self.freeze_backbone = (
            freeze_backbone  # True only when backbone is frozen (stage 2)
        )
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        raw = getattr(model, "_orig_mod", model)
        self.updates += 1
        d = self.decay(self.updates)
        msd = raw.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            mv = msd[k].detach().to(v.dtype)
            if self.freeze_backbone and k.startswith("backbone."):
                v.copy_(mv)  # frozen FISTA: track exactly
            else:
                v.mul_(d).add_(mv, alpha=1 - d)  # EMA normally

    def state_dict(self):
        return self.ema.state_dict()


# In[65]:


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


def save_ckpt(path, model, opt, sched, ep, stage, tag, ap50=None, ema=None):
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
            "ema": ema.state_dict() if ema is not None else None,
        },
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


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


# In[66]:


import math


def _sgd(model, cfg, epochs, lr=None):
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
    lr = cfg.LR_INIT if lr is None else lr
    params = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.SGD(
        params,
        lr=lr,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        nesterov=True,
    )

    warmup = cfg.WARMUP_EPOCHS
    total_steps = epochs

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / (total_steps - warmup)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return (cfg.MIN_LR / lr) + (1 - cfg.MIN_LR / lr) * cosine_decay

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


# In[67]:


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
    ema=None,
    track_best_after=0,
):
    """One training stage. All checkpoints + the eval CSV are written into out_dir.
    do_eval=False skips validation (used for smoke tests).
    track_best_after: ignore this many initial epochs before updating best_path
        (avoids locking in an early high-LR AP50 spike as 'best')."""
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

            if not torch.isfinite(loss):  # skip non-finite loss
                opt.zero_grad(set_to_none=True)
                del loss
                nb += 1
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=10.0
            )
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

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
            save_ckpt(ckpt_path, model, opt, sched, ep, stage, tag, ema=ema)

        if do_eval and ((ep + 1) % eval_every == 0 or (ep + 1) == epochs):
            eval_target = ema.ema if ema is not None else model
            ap50, f1 = evaluate(
                eval_target,
                val_loader,
                device,
                use_dam,
                strides=[8, 16, 32],
                num_classes=1,
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
            track_ok = (ep + 1) > track_best_after
            if best_path is not None and track_ok and ap50 * 100 > best_ap50:
                best_ap50 = ap50 * 100
                # save the weights that were actually evaluated (EMA if present)
                save_ckpt(
                    best_path,
                    eval_target,
                    opt,
                    sched,
                    ep,
                    stage,
                    tag,
                    ap50=best_ap50,
                    ema=ema,
                )
                print(f"[{tag}] *** new best AP50 {best_ap50:.2f} -> {best_path}")
            elif best_path is not None and not track_ok:
                print(f"[{tag}] (best tracking starts after epoch {track_best_after})")

    if final_path is not None:
        deploy = ema.ema if ema is not None else model
        save_ckpt(final_path, deploy, opt, sched, epochs - 1, stage, tag, ema=ema)
        print(f"[{tag}] saved stage checkpoint -> {final_path}")


def _filtered_load(module, sd):
    """Load only keys that exist in `module` with a matching shape.
    Returns the number of source keys dropped (absent or shape-mismatched)."""
    tgt = module.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt and v.shape == tgt[k].shape}
    module.load_state_dict(filt, strict=False)
    return len(sd) - len(filt)


def _load_into(path, model, opt=None, sched=None, ema=None):
    """Resume-load a checkpoint. Model/EMA loaded by shape-matched keys (so stale or
    re-architected disp.* params are skipped). opt/sched best-effort.
    Returns the epoch to resume from (stored epoch + 1), or 0 if the file is absent."""
    if not os.path.exists(path):
        return 0
    ck = torch.load(path, map_location="cpu")
    raw = getattr(model, "_orig_mod", model)

    dropped = _filtered_load(raw, _strip_compile(ck["model"]))
    if dropped:
        print(
            f"[resume] {path}: skipped {dropped} incompatible key(s) (e.g. stale disp.*)"
        )

    if opt is not None and ck.get("opt"):
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            print(f"[resume] optimizer state not restored ({e})")
    if sched is not None and ck.get("sched"):
        try:
            sched.load_state_dict(ck["sched"])
        except Exception as e:
            print(f"[resume] scheduler state not restored ({e})")
    if ema is not None and ck.get("ema"):
        _filtered_load(ema.ema, ck["ema"])  # weights only; decay ramp restarts

    nxt = ck.get("epoch", -1) + 1
    print(f"[resume] loaded {path}  -> resume at epoch {nxt}")
    return nxt


def _seed_backbone_from_fista_best(model, out_dir):
    """Seed stage 2 from the BEST FISTA checkpoint (frozen backbone + pool/fpn/head).

    The backbone is FROZEN in stage 2, so whatever weights are loaded here are locked
    in for the whole run -> use the best FISTA EMA weights, not the last-epoch ones.

    disp.* is stripped from the source state_dict so the zero-init DAM branch (fix #2)
    is NOT overwritten by stage-1's untrained random disp weights.

    Falls back to fista.pth (final EMA deploy) if fista_best.pth is missing.
    No-op if neither exists (e.g. resuming an in-progress stage 2 from dam_last.pth).
    """
    for name, kind in (("fista_best.pth", "best"), ("fista.pth", "final (fallback)")):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu")
            sd = {
                k: v
                for k, v in _strip_compile(ck["model"]).items()
                if not k.startswith("disp.")  # keep DAM at zero-init (fix #2)
            }
            raw = getattr(model, "_orig_mod", model)
            dropped = _filtered_load(raw, sd)
            print(
                f"[stage2] seeded backbone/pool/fpn/head from FISTA {kind}: {path}  "
                f"(disp.* kept at zero-init; {dropped} non-matching keys skipped)"
            )
            return
    print("[stage2] no FISTA checkpoint found to seed from — using current weights")


def train_mocid(cfg, tag="default", train_loader=None, val_loader=None, do_eval=True):
    """Two-stage training with auto-resume from runs/<tag>/.
    Resumes stage 1 or stage 2 from *_last.pth if present; skips a completed stage."""
    model = MOCID(num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)
    model = torch.compile(model)
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
    ema = ModelEMA(model)
    s1_start = _load_into(
        os.path.join(out_dir, "fista_last.pth"), model, opt, sched, ema
    )
    if s1_start < cfg.EPOCHS_SPTBACKBONE:
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
            start_ep=s1_start,
            ckpt_path="fista_last.pth",
            final_path="fista.pth",
            best_path="fista_best.pth",
            csv_path="eval_log.csv",
            eval_every=cfg.EVAL_EVERY,
            do_eval=do_eval,
            ema=ema,
            track_best_after=cfg.TRACK_BEST_AFTER,
        )
    else:
        print(
            f"[train] stage 1 already complete ({s1_start} epochs) — "
            "backbone will be seeded from BEST FISTA in stage 2"
        )

    # ---------- Stage 2: freeze FISTA -> train DAM ----------
    print("=" * 25, "STAGE 2: DAM", "=" * 25)
    set_stage(model, 2)

    # Seed the frozen backbone (+ pool/fpn/head) from the BEST FISTA weights, not last.
    # Handles BOTH paths: continuous (fista->dam, overrides live ep100 weights) and
    # separate (fista done -> stop -> dam). disp.* is stripped so fix #2 survives.
    # Runs BEFORE ModelEMA(...) so the EMA snapshot also starts from the best backbone.
    _seed_backbone_from_fista_best(model, out_dir)

    opt, sched = _sgd(model, cfg, cfg.EPOCHS_DAM, lr=cfg.LR_DAM)  # <-- lower LR
    # fast EMA for stage 2: a fresh disp branch must be TRACKED, not smoothed
    # over ~10k updates. decay 0.999 / tau 300 -> ~1k-update window.
    ema = ModelEMA(model, decay=0.999, tau=300, freeze_backbone=True)
    s2_start = _load_into(os.path.join(out_dir, "dam_last.pth"), model, opt, sched, ema)
    if s2_start < cfg.EPOCHS_DAM:
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
            start_ep=s2_start,
            ckpt_path="dam_last.pth",
            final_path="dam.pth",
            best_path="dam_best.pth",
            csv_path="eval_log.csv",
            eval_every=cfg.EVAL_EVERY,
            do_eval=do_eval,
            ema=ema,
        )
    else:
        print(f"[train] stage 2 already complete ({s2_start} epochs)")
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
