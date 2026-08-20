"""
Convert ITSDT-15k (Pascal-VOC XML) -> YOLO detection format for TY-RIST.
Single class, official sequence split (1-76 train, 77-87 val). Evaluation/ MATLAB
scorer is ignored (paper reports mAP50/F1). Emits a lowercase images/+labels/ tree
via symlinks; writes configs/data/itsdt_15k.yaml with a CWD-relative path.
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

# --- paths anchored to this script, so it runs from any CWD -------------
SCRIPT_DIR  = Path(__file__).resolve().parent              # <repo>/convert
CODE_ROOT   = SCRIPT_DIR.parent                            # <repo>  (= /TyRST)
DATASET_DIR = (CODE_ROOT / ".." / ".." / "datasets").resolve()   # ../../datasets

DATASET_ROOT = DATASET_DIR / "ITSDT-15k"
YAML_OUT     = CODE_ROOT / "configs" / "data" / "itsdt_15k.yaml"
OUT_ROOT     = DATASET_ROOT / "yolo"

IMAGES_DIRNAME = "Images"
TRAIN_SEQ_MAX = 76
IMG_EXT = ".bmp"
# ----------------------------------------------------------------------


def find_annotation_dir(root: Path) -> Path:
    for name in ("Annotation", "Annotations"):
        if (root / name).is_dir():
            return root / name
    raise FileNotFoundError(f"No Annotation/Annotations dir under {root}")


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width"))  if size is not None and size.findtext("width")  else None
    h = int(size.findtext("height")) if size is not None and size.findtext("height") else None
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        boxes.append((float(b.findtext("xmin")), float(b.findtext("ymin")),
                      float(b.findtext("xmax")), float(b.findtext("ymax"))))
    return w, h, boxes


def to_yolo(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, min(xmin, w)), max(0.0, min(xmax, w))
    ymin, ymax = max(0.0, min(ymin, h)), max(0.0, min(ymax, h))
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    return ((xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, bw / w, bh / h)


def relink(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    images_root = DATASET_ROOT / IMAGES_DIRNAME
    ann_root = find_annotation_dir(DATASET_ROOT)
    assert images_root.is_dir(), f"missing {images_root}"

    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted((d for d in images_root.iterdir() if d.is_dir()),
                      key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30)

    stats = dict(frames=0, train=0, val=0, with_obj=0, empty=0, boxes=0,
                 skipped_box=0, missing_xml=0, missing_size=0)

    for seq in seq_dirs:
        if not seq.name.isdigit():
            print(f"  ! skipping non-numeric sequence dir: {seq.name}")
            continue
        split = "train" if int(seq.name) <= TRAIN_SEQ_MAX else "val"

        for img in sorted(seq.glob(f"*{IMG_EXT}")):
            stats["frames"] += 1
            stats[split] += 1
            stem = f"{seq.name}_{img.stem}"

            xml = ann_root / seq.name / f"{img.stem}.xml"
            lines = []
            if xml.is_file():
                w, h, boxes = parse_xml(xml)
                if not (w and h):
                    from PIL import Image
                    with Image.open(img) as im:
                        w, h = im.size
                    stats["missing_size"] += 1
                for box in boxes:
                    yb = to_yolo(box, w, h)
                    if yb is None:
                        stats["skipped_box"] += 1
                        continue
                    lines.append(f"0 {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")
            else:
                stats["missing_xml"] += 1

            stats["with_obj" if lines else "empty"] += 1
            stats["boxes"] += len(lines)

            relink(img, OUT_ROOT / "images" / split / f"{stem}{IMG_EXT}")
            (OUT_ROOT / "labels" / split / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "")

    rel_path = Path(os.path.relpath(OUT_ROOT, CODE_ROOT)).as_posix()   # ../../datasets/ITSDT-15k/yolo
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    YAML_OUT.write_text(
        f"# ITSDT-15k (multi-frame, evaluated single-frame; TY-RIST Sec 4.3/5.1)\n"
        f"# Official split: sequences 1-{TRAIN_SEQ_MAX} train, {TRAIN_SEQ_MAX+1}-87 val.\n"
        f"# path is relative to the repo root (/TyRST); run training from there.\n"
        f"path: {rel_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/val\n"
        f"nc: 1\n"
        f"names: [target]\n"
    )

    print("\n==== ITSDT-15k conversion summary ====")
    for k, v in stats.items():
        print(f"  {k:12s}: {v}")
    print(f"  yaml written : {YAML_OUT}  (path: {rel_path})")
    print(f"  data root    : {OUT_ROOT}")


if __name__ == "__main__":
    main()