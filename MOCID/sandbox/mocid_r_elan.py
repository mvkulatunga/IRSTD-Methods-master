#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    BATCH_SIZE = 8

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

# In[2]:


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

# In[3]:


from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import A2C2f, ELAN1


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

# In[4]:


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

# In[5]:


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

# In[6]:


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

# In[7]:


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
        return f + f_out


# <img src="./diagrams/fista_layer_architecture.png" alt="fista_layer" style="max-width: 50%;">

# In[8]:


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

# In[9]:


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

        # ---------------------------------------------------------
        # CHANGED: Replaced CSPLayer with ELAN1 for shallow layers
        # ---------------------------------------------------------
        self.spatial_layer1 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, ksize=3, stride=2),
            ELAN1(
                c1=base_channels * 2,
                c2=base_channels * 2,
                c3=base_channels * 2,
                c4=base_channels,
            ),
        )
        self.spatial_layer2 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, ksize=3, stride=2),
            ELAN1(
                c1=base_channels * 4,
                c2=base_channels * 4,
                c3=base_channels * 4,
                c4=base_channels * 2,
            ),
        )
        # ---------------------------------------------------------

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
        s = self.stem(frame)
        s = self.spatial_layer1(s)
        s = self.spatial_layer2(s)
        return s

    def _downsample_clip(self, conv, clip):
        B, T, C, H, W = clip.shape
        out = conv(clip.reshape(B * T, C, H, W))
        _, Co, Ho, Wo = out.shape
        return out.view(B, T, Co, Ho, Wo)

    def _run_fista(self, clip):
        c = self._downsample_clip(self.downsample1, clip)
        o1 = self.fista_layer1(c)
        c = self._downsample_clip(self.downsample2, o1)
        o2 = self.fista_layer2(c)
        c = self._downsample_clip(self.downsample3, o2)
        o3 = self.fista_layer3(c)
        return o1, o2, o3

    def forward(self, x):
        B, T, C, H, W = x.shape  # T == 5
        s = [self._encode_spatial(x[:, i]) for i in range(T)]
        clip = torch.stack(s, dim=1)  # (B, 5, Cf, Hf, Wf)

        o1, o2, o3 = self._run_fista(clip)  # each (B, 5, C_k, H_k, W_k)

        Ft = [o1[:, -1], o2[:, -1], o3[:, -1]]  # 3 maps, each (B, C, H, W)
        Fr_list = [o1[:, :-1], o2[:, :-1], o3[:, :-1]]  # 3 maps, each (B, 4, C, H, W)

        return Ft, Fr_list


# YoloXHead for Detection
# ```tex
# Ge, Zheng & Liu, Songtao & Wang, Feng & Li, Zeming & Sun, Jian. (2021). YOLOX: Exceeding YOLO Series in 2021. 10.48550/arXiv.2107.08430. 
# ```
# IOULoss, YoloLoss and YoloPAFPN taken directly from [SSTnet](https://github.com/UESTC-nnLab/SSTNet) which has taken it from [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)

# In[10]:


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


# In[11]:


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


# In[12]:


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
            topk_ious.sum(1).int(), min=1, max=pair_wise_ious.size(1)
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


# In[13]:


class YOLOPAFPN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        c3, c4, c5 = ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.l5 = BaseConv(c5, c4, 1, 1)

        # ---------------------------------------------------------
        # CHANGED: Replaced CSPLayer with A2C2f for deeper layers.
        # a2=False turns off area-attention and defaults to C3k blocks.
        # ---------------------------------------------------------
        self.p4 = A2C2f(2 * c4, c4, n=1, a2=False)
        self.l4 = BaseConv(c4, c3, 1, 1)
        self.p3 = A2C2f(2 * c3, c3, n=1, a2=False)

        self.d3 = BaseConv(c3, c3, 3, 2)
        self.n4 = A2C2f(2 * c3, c4, n=1, a2=False)

        self.d4 = BaseConv(c4, c4, 3, 2)
        self.n5 = A2C2f(2 * c4, c5, n=1, a2=False)
        # ---------------------------------------------------------

    def forward(self, xT, ff=None):
        x3, x4, x5 = xT
        if ff is not None:
            x3 = x3 + ff[0]
            x4 = x4 + ff[1]
            x5 = x5 + ff[2]

        a = self.l5(x5)
        p4 = self.p4(torch.cat([self.up(a), x4], 1))
        b = self.l4(p4)
        p3 = self.p3(torch.cat([self.up(b), x3], 1))

        n4 = self.n4(torch.cat([self.d3(p3), b], 1))
        n5 = self.n5(torch.cat([self.d4(n4), a], 1))
        return [p3, n4, n5]


# <img src="./diagrams/mocid_architecture.png" alt="mocid" style="max-height: 200px;">
# 

# In[14]:


class TemporalPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats_by_scale):
        pooled = []
        for scale_feat in feats_by_scale:
            # Pool across the Time dimension (dim=1)
            pooled.append(scale_feat.amax(dim=1))
        return pooled


class MOCID(nn.Module):
    def __init__(
        self, num_classes=1, num_frames=5, img_size=512, base_channels=16, d_state=16
    ):
        super().__init__()
        ch = [base_channels * 8, base_channels * 16, base_channels * 32]
        self.backbone = SpatioTemporalBackbone(3, base_channels, num_frames, img_size)
        self.pool = TemporalPooling()
        # self.disp = DisplacementNet(ch, d_state)
        self.fpn = YOLOPAFPN(ch)
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

        # if use_dam:
        #     # DAM (Mamba selective-scan) is unstable in fp16 -> run it in fp32
        #     with torch.autocast("cuda", enabled=False):
        #         motion_features = self.disp([f.float() for f in feats_by_scale])
        # else:
        motion_features = feats_by_scale

        F_f = self.pool(motion_features)
        for t in F_f:
            assert torch.isfinite(t).all(), "non-finite in F_f (DAM output)"
        outs = self.head(self.fpn(F_T, F_f))

        if labels is not None:
            with torch.amp.autocast("cuda", enabled=False):
                return self.loss_fn([o.float() for o in outs], labels)
        return outs


# In[15]:


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


# In[ ]:


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


# In[ ]:


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
    start_ep=0,
    ckpt_every=5,
    ckpt_path="ckpt_last.pth",
    final_path=None,
    best_path=None,
    eval_every=20,
    csv_path="eval_log.csv",
):
    """Runs the training loop for one stage: trains for `epochs` epochs,
    periodically checkpointing, evaluating, and logging results.

    Args:
        model (nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        opt (torch.optim.Optimizer): Optimizer.
        sched (torch.optim.lr_scheduler._LRScheduler): LR scheduler (stepped once per epoch).
        epochs (int): Total number of epochs for this stage.
        device (str): Device to run on (e.g. "cuda").
        use_dam (bool): Whether to enable the DAM branch in the model forward pass.
        tag (str): Run tag, used in logging and checkpoint metadata.
        stage (int): Training stage index, stored in checkpoints.
        start_ep (int): Epoch to resume from. Defaults to 0.
        ckpt_every (int): Save a "last" checkpoint every N epochs. Defaults to 5.
        ckpt_path (str): Path for the rolling "last" checkpoint. Defaults to "ckpt_last.pth".
        final_path (str, optional): Path to save a final checkpoint at stage end. Defaults to None.
        best_path (str, optional): Path to save the best-AP50 checkpoint. Defaults to None.
        eval_every (int): Run validation every N epochs. Defaults to 20.
        csv_path (str): Path to the evaluation CSV log. Defaults to "eval_log.csv".

    Returns:
        None
    """

    scaler = torch.amp.GradScaler("cuda")

    # seed best from an existing best checkpoint so a resume can't clobber it
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
            model.backbone.eval()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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

        if (ep + 1) % eval_every == 0 or (ep + 1) == epochs:
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

            # --- save best-by-AP50 ---
            if best_path is not None and ap50 * 100 > best_ap50:
                best_ap50 = ap50 * 100
                save_ckpt(best_path, model, opt, sched, ep, stage, tag, ap50=best_ap50)
                print(f"[{tag}] *** new best AP50 {best_ap50:.2f} -> {best_path}")

    if final_path is not None:
        save_ckpt(final_path, model, opt, sched, epochs - 1, stage, tag)
        print(f"[{tag}] saved stage checkpoint -> {final_path}")


# In[ ]:


def train_mocid(cfg, device="cuda", tag="default"):
    """Trains the MOCID model (FISTA only, NO DAM).
    Supports resuming from the last saved checkpoint.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = os.path.join("runs", tag)
    os.makedirs(run_dir, exist_ok=True)
    p = lambda name: os.path.join(run_dir, name)
    resume, csv_path = p("ckpt_last.pth"), p("eval_log.csv")
    print(f"[run] tag={tag} -> {run_dir}")

    train_ds = MOCIDDataset(
        cfg.train_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=True
    )
    val_ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)
    assert len(train_ds) > 0 and len(val_ds) > 0

    loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_mocid,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_eval,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = MOCID(num_classes=1, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    ckpt = torch.load(resume, map_location="cpu") if os.path.exists(resume) else None
    if ckpt:
        getattr(model, "_orig_mod", model).load_state_dict(
            _strip_compile(ckpt["model"])
        )
        _restore_rng(ckpt)
        print(f"resuming from stage {ckpt['stage']} epoch {ckpt['epoch']+1}")

    # --- SINGLE STAGE: Train Backbone (FISTA ONLY) ---
    opt, sched = _sgd(model, cfg, cfg.EPOCHS_SPTBACKBONE)
    start = 0
    if ckpt and ckpt["stage"] == 1:
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start = ckpt["epoch"] + 1

    _run(
        model,
        loader,
        val_loader,
        opt,
        sched,
        cfg.EPOCHS_SPTBACKBONE,
        device,
        use_dam=False,
        tag="STB",
        stage=1,
        start_ep=start,
        ckpt_path=resume,
        final_path=p("ckpt_stage1.pth"),
        best_path=p("ckpt_stage1_best.pth"),
        eval_every=cfg.EVAL_EVERY,
        csv_path=csv_path,
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

# 

# 

# 

# 

# 

# AI Generated Helpers (for mamba base, and 3DCDC)

# In[ ]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        selective_scan_ref,
    )
except ImportError:
    selective_scan_fn = None

    def selective_scan_ref(u, delta, A, B, C, delta_softplus=True, **kw):
        if delta_softplus:
            delta = F.softplus(delta)
        b, d, L = u.shape
        dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        dB = torch.einsum("bdl,bnl->bdln", delta, B)
        h = u.new_zeros(b, d, A.shape[1])
        ys = []
        for i in range(L):
            h = dA[:, :, i] * h + dB[:, :, i] * u[:, :, i : i + 1]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, i]))
        return torch.stack(ys, -1)


def _ssm(*a, **k):
    fn = (
        selective_scan_fn
        if (selective_scan_fn and a[0].is_cuda)
        else selective_scan_ref
    )
    return fn(*a, **k)


# In[ ]:


class CDC3d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=(2, 3, 3),
        stride=1,
        padding=(0, 1, 1),
        bias=False,
        theta=0.7,
        center=(1, 1, 1),
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.theta = theta
        self.center = center

    def forward(self, x):
        w = self.conv.weight
        w_sum = w.sum(dim=(2, 3, 4), keepdim=True)
        mask = torch.zeros_like(w)
        ct, ch, cw = self.center
        mask[:, :, ct, ch, cw] = 1.0
        cdc_w = w - self.theta * w_sum * mask
        return F.conv3d(x, cdc_w, self.conv.bias, self.conv.stride, self.conv.padding)


# <img src="./diagrams/sds_architecture.svg" alt="sds_architecture" style="max-width: 50%; height: 400px">
# 
# 
# SDS models the fine-grained spatio-temporal differences between the target frame and a reference frame to generate input-dependent system parameters for Mamba.
# 
# $$x = \text{Concat}(F_{T}, F_{R})$$
# $$B, C, \Delta = \text{3DCDC}(x)$$
# $$\overline{A}, \overline{B}, \overline{C} = \text{Discretize}(A, B, C, \Delta)$$
# 
# * $F_{T}, F_{R} \in \mathbb{R}^{C \times H \times W}$ are the target and reference frames.
# * $x \in \mathbb{R}^{2 \times C \times H \times W}$ is the temporally concatenated sequence.
# * $\text{3DCDC}$ denotes the 3D Central Difference Convolution.
# * $\overline{A}, \overline{B}, \overline{C}$ are the discretized state-space parameters used for scanning.

# In[ ]:


class SDS(nn.Module):
    def __init__(
        self, d_inner, d_state=16, theta=0.7, dt_rank=None, dt_min=1e-3, dt_max=1e-1
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)

        self.cdc3d = CDC3d(
            d_inner,
            self.dt_rank + 2 * d_state,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1),
            theta=theta,
            center=(1, 1, 1),
        )

        self.dt_proj = nn.Conv2d(self.dt_rank, d_inner, 1, bias=True)
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

    @property
    def A(self):
        return -torch.exp(self.A_log.float())

    def forward(self, F_T, F_R):
        out = self.cdc3d(torch.stack([F_R, F_T], dim=2)).squeeze(2)
        dt, Bp, Cp = torch.split(out, [self.dt_rank, self.d_state, self.d_state], dim=1)
        delta = self.dt_proj(dt)
        return (delta, Bp, Cp)


# <img src="./diagrams/tis_architecture.svg" alt="tis_architecture" style="max-width: 50%; height: 400px">
# 
# 
# TIS constructs displacement-aware sequences by interpolating pooled features, which are then scanned using the parameters derived from SDS.
# 
# $$\overline{F}_{T} = SP(F_{T}), \quad \overline{F}_{R} = SP(F_{R})$$
# $$X_{W} = \text{Interpolation}(\overline{F}_{T}, \overline{F}_{R})$$
# $$Y_{W} = \text{Scan}(X_{W}, \overline{A}, \overline{B}, \overline{C})$$
# 
# * $SP$ denotes spatial pooling (e.g., $1 \times 2$ for width, $2 \times 1$ for height).
# * $X_{W} \in \mathbb{R}^{L \times C}$ is the width-interpolated sequence interleaving reference and target tokens.
# * $\text{Scan}$ represents the bidirectional selective scan mechanism. (This process is repeated symmetrically for the height dimension $X_{H}$).

# In[ ]:


class TIS(nn.Module):
    @staticmethod
    def _pool(p, k):
        *lead, H, W = p.shape
        x = F.avg_pool2d(p.reshape(-1, 1, H, W), k)
        return x.reshape(*lead, x.shape[-2], x.shape[-1])

    @staticmethod
    def _il(r, t):
        return torch.stack([r, t], dim=-1).reshape(*r.shape[:-1], -1)

    def _scan(self, u, delta, A, B, C):
        yf = _ssm(u, delta, A, B, C, delta_softplus=True)
        yb = _ssm(
            u.flip(-1), delta.flip(-1), A, B.flip(-1), C.flip(-1), delta_softplus=True
        ).flip(-1)
        return yf + yb

    def _branch(self, xT, xR, p, A, k):
        d_, B_, C_ = p
        Bb, d, H, W = xT.shape
        u = self._il(self._pool(xR, k).flatten(2), self._pool(xT, k).flatten(2))
        sh = lambda t: self._il(
            self._pool(t, k).flatten(-2), self._pool(t, k).flatten(-2)
        )
        y = self._scan(u, sh(d_), A, sh(B_), sh(C_))
        return y.reshape(Bb, d, H, W)

    def forward(self, xT, xR, p, A):
        return self._branch(xT, xR, p, A, (1, 2)) + self._branch(xT, xR, p, A, (2, 1))


# <img src="./diagrams/dam_architecture.svg" alt="dam_architecture" style="max-width: 50%; height: 400px">
# 
# 
# DAM integrates the input projections, the TIDS (SDS + TIS) mechanism, and the output projections to enhance the target frame features using displacement information.
# 
# $$x_{T}, z_{T} = \text{Linear}(F_{T})$$
# $$x_{R}, \_ = \text{Linear}(F_{R})$$
# $$y = \text{TIDS}(x_{T}, x_{R}) \odot \text{SiLU}(z_{T})$$
# 
# * $x_{T}, z_{T}$ are the projected state and gate features for the target frame.
# * $\text{TIDS}$ is (SDS + TIS).
# * $\odot$ represents element-wise multiplication with the activation gate.

# In[ ]:


class DAM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=3, theta=0.7):
        super().__init__()
        m = int(expand * d_model)
        self.d_inner = m
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * m, bias=False)
        self.dwconv = nn.Conv2d(m, m, d_conv, padding=d_conv // 2, groups=m)
        self.sds = SDS(m, d_state, theta)
        self.tis = TIS()
        self.out_proj = nn.Linear(m, d_model, bias=False)

    def _proj(self, feat):
        h = self.norm(feat.permute(0, 2, 3, 1))
        x, z = self.in_proj(h).chunk(2, dim=-1)
        return x.permute(0, 3, 1, 2), z.permute(0, 3, 1, 2)

    def forward(self, feat_t, feat_r):
        xT, zT = self._proj(feat_t)
        xR, _ = self._proj(feat_r)

        xT = F.silu(self.dwconv(xT))
        xR = F.silu(self.dwconv(xR))

        p = self.sds(xT, xR)
        y = self.tis(xT, xR, p, self.sds.A) * F.silu(zT)

        return feat_t + self.out_proj(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# <img src="./diagrams/displacement_model_architecture.svg" alt="displacement_modeling_architecture" style="max-width: 50%; height: 400px">
# 
# 
# This network applies the shared DAM across all reference frames relative to the target frame and fuses them using Temporal Pooling.
# 
# $$F_{f} = (\{\text{DAM}(F_{T}, F_{i})\}_{i=1}^{T-1})$$
# 
# * $F_{T}$ is the target frame (typically the last frame at index $T$).
# * $F_{i}$ represents each historical reference frame in the clip.

# In[ ]:


class DisplacementNet(nn.Module):
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.dams = nn.ModuleList([DAM(c, d_state) for c in channels])

    @torch.compiler.disable
    def forward(self, feats_by_scale):
        f_f_unpooled = []
        for s, f in enumerate(feats_by_scale):
            F_T = f[:, -1]

            # compute the DAM outputs and stack them (shape: B, T-1, C, H, W)
            F_f = torch.stack(
                [self.dams[s](F_T, f[:, i]) for i in range(f.shape[1] - 1)], dim=1
            )
            f_f_unpooled.append(F_f)

        return f_f_unpooled

