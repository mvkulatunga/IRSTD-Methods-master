"""
Convert IRDST (real + simulation) -> YOLO detection format for TY-RIST.
Box format is [x_topleft, y_topleft, w, h] in pixels (verified vs center/).
real+sim merged; each source's train/ -> train, test/ -> val. Emits lowercase
images/+labels/ via symlinks; writes configs/data/irdst.yaml with a relative path.
"""

import os
from pathlib import Path
from PIL import Image

# --- paths anchored to this script, so it runs from any CWD -------------
SCRIPT_DIR  = Path(__file__).resolve().parent              # <repo>/convert
CODE_ROOT   = SCRIPT_DIR.parent                            # <repo>  (= /TyRST)
DATASET_DIR = (CODE_ROOT / ".." / ".." / "datasets").resolve()   # ../../datasets

YAML_OUT = CODE_ROOT / "configs" / "data" / "irdst.yaml"
OUT_ROOT = DATASET_DIR / "IRDST_yolo"
SOURCES = {
    "real": DATASET_DIR / "IRDST_real",
    "sim":  DATASET_DIR / "IRDST_simulation",
}
SPLIT_MAP = {"train": "train", "test": "val"}
IMG_EXT = ".png"
BOX_IS_TOPLEFT = True
# ----------------------------------------------------------------------


def parse_boxes(txt_path: Path):
    boxes = []
    for line in txt_path.read_text().splitlines():
        s = line.strip().strip("[]").replace(",", " ")
        if not s:
            continue
        nums = [float(t) for t in s.split()]
        if len(nums) >= 4:
            boxes.append(tuple(nums[:4]))
    return boxes


def to_yolo(box, W, H):
    x, y, w, h = box
    if not BOX_IS_TOPLEFT:
        x, y = x - w / 2.0, y - h / 2.0
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    return ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, bw / W, bh / H)


def relink(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = dict(frames=0, train=0, val=0, with_obj=0, empty=0, boxes=0,
                 skipped_box=0, missing_box=0)

    for prefix, root in SOURCES.items():
        if not root.is_dir():
            print(f"  ! source missing, skipping: {root}")
            continue
        for split_src, split_dst in SPLIT_MAP.items():
            img_dir = root / "images" / split_src
            box_dir = root / "boxes" / split_src
            if not img_dir.is_dir():
                print(f"  ! {img_dir} not found, skipping")
                continue

            for img in sorted(img_dir.glob(f"*{IMG_EXT}")):
                stats["frames"] += 1
                stats[split_dst] += 1
                stem = f"{prefix}_{img.stem}"

                box_file = box_dir / f"{img.stem}.txt"
                lines = []
                if box_file.is_file():
                    with Image.open(img) as im:
                        W, H = im.size
                    for box in parse_boxes(box_file):
                        yb = to_yolo(box, W, H)
                        if yb is None:
                            stats["skipped_box"] += 1
                            continue
                        lines.append(f"0 {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")
                else:
                    stats["missing_box"] += 1

                stats["with_obj" if lines else "empty"] += 1
                stats["boxes"] += len(lines)

                relink(img, OUT_ROOT / "images" / split_dst / f"{stem}{IMG_EXT}")
                (OUT_ROOT / "labels" / split_dst / f"{stem}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "")

    rel_path = Path(os.path.relpath(OUT_ROOT, CODE_ROOT)).as_posix()   # ../../datasets/IRDST_yolo
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    YAML_OUT.write_text(
        f"# IRDST (real + simulation merged; multi-frame, TY-RIST Sec 4.3/5.1)\n"
        f"# val = each source's provided test split.\n"
        f"# path is relative to the repo root (/TyRST); run training from there.\n"
        f"path: {rel_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/val\n"
        f"nc: 1\n"
        f"names: [target]\n"
    )

    print("\n==== IRDST conversion summary ====")
    for k, v in stats.items():
        print(f"  {k:12s}: {v}")
    print(f"  yaml written : {YAML_OUT}  (path: {rel_path})")
    print(f"  data root    : {OUT_ROOT}")


if __name__ == "__main__":
    main()