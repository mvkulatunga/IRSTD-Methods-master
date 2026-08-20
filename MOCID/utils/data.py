import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MOCIDDataset(Dataset):
    """T-frame clips from an annotation file; the last frame carries the labels."""

    def __init__(self, annotations_file, T=5, img_size=(512, 512), is_train=True):
        """annotations_file lines are '<img_path> xmin,ymin,xmax,ymax,cls [...]'."""
        self.T = T
        self.img_size = img_size
        self.is_train = is_train
        self.clips = []

        if not os.path.exists(annotations_file):
            print(f"Warning: {annotations_file} not found.")
            return

        # seq_id -> {frame_num -> {"path", "boxes"}}, accumulating boxes per frame
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
                    frame["boxes"].extend(boxes)  # same frame listed again

        # slide a T-wide window per sequence, dropping windows that span a gap
        for seq_id, frame_map in sequences.items():
            frames = [
                {"frame_num": fn, "path": fr["path"], "boxes": fr["boxes"]}
                for fn, fr in sorted(frame_map.items())
            ]
            for i in range(self.T - 1, len(frames)):
                window = frames[i - (self.T - 1) : i + 1]
                if window[-1]["frame_num"] - window[0]["frame_num"] != self.T - 1:
                    continue
                self.clips.append(window)

        print(
            f"Loaded {len(self.clips)} clips of length {self.T} from {annotations_file}"
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """idx -> ((T,3,H,W) float clip in [0,1], {"boxes": (N,4) xyxy, "labels": (N,)})."""
        clip_data = self.clips[idx]
        target_info = clip_data[-1]

        sample_img = cv2.imread(clip_data[0]["path"], cv2.IMREAD_COLOR)
        if sample_img is None:
            raise FileNotFoundError(f"Could not read image at {clip_data[0]['path']}")
        orig_h, orig_w = sample_img.shape[:2]
        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h

        do_flip = self.is_train and (random.random() > 0.5)

        # rescale target-frame boxes into img_size space, mirroring x on flip
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

        # the flip is shared across the clip so temporal consistency holds
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


def collate_train(batch):
    """batch -> ((B,T,3,H,W) clips, list of (n_gt,5) [cx,cy,w,h,cls] label tensors)."""
    clips = torch.stack([b[0] for b in batch])
    labels = []
    for _, tgt in batch:
        bx = tgt["boxes"]
        cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2
        w, h = bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]
        cls = tgt["labels"].float()
        labels.append(torch.stack([cx, cy, w, h, cls], dim=1))
    return clips, labels


def collate_eval(batch):
    """batch -> ((B,T,3,H,W) clips, list of (n_gt,4) xyxy GT tensors)."""
    clips = torch.stack([b[0] for b in batch])
    gts = [b[1]["boxes"].clone() for b in batch]
    return clips, gts
