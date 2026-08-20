#!/usr/bin/env python3
"""Reorganize raw DAUB zips into the SSTNet/MOCID layout, driven by the annotation files."""
import argparse, os, re, shutil, sys
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

LINE_RE = re.compile(r'^(.*?/images/(train|test)/(data\d+)/(\d+)\.bmp)\s+(.*)$')

def find_seq_dir(raw: Path, seq: str) -> Path:
    """Raw layout is usually DAUB/dataN/dataN/*.bmp; fall back to DAUB/dataN/."""
    nested = raw / seq / seq
    flat   = raw / seq
    if nested.is_dir() and any(nested.glob("*.bmp")): return nested
    if flat.is_dir()   and any(flat.glob("*.bmp")):   return flat
    raise FileNotFoundError(f"No .bmp frames found for {seq} under {raw}")

def raw_frame_map(seq_dir: Path):
    """Map integer stem -> path, plus a numerically sorted list for positional fallback."""
    m = {}
    for p in seq_dir.glob("*.bmp"):
        try: m[int(p.stem)] = p
        except ValueError: pass
    ordered = [m[k] for k in sorted(m)]
    return m, ordered

def place(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    if mode == "copy":     shutil.copy2(src, dst)
    elif mode == "link":   os.link(src, dst)      # hardlink (same filesystem)
    elif mode == "symlink":os.symlink(src.resolve(), dst)

def process(ann_file: Path, raw: Path, out: Path, mode: str, dry: bool):
    # group annotation lines by sequence, preserving order
    by_seq = defaultdict(list)   # seq -> list of (split, idx, bbox_str)
    for line in ann_file.read_text().splitlines():
        line = line.strip()
        if not line: continue
        mt = LINE_RE.match(line)
        if not mt:
            print(f"  [skip] unparsable: {line[:80]}"); continue
        _, split, seq, idx, bbox = mt.groups()
        by_seq[seq].append((split, int(idx), bbox))

    rewritten, missing = [], 0
    for seq, items in by_seq.items():
        seq_dir = find_seq_dir(raw, seq)
        name_map, ordered = raw_frame_map(seq_dir)
        need = sorted(i for _, i, _ in items)
        use_positional = not all(i in name_map for i in need)
        if use_positional:
            print(f"  [warn] {seq}: exact-name match failed (raw likely not 0-indexed); "
                  f"falling back to positional alignment of {len(need)} labeled frames.")

        for split, idx, bbox in tqdm(items, desc=f"  {seq}", leave=False):
            if not use_positional:
                src = name_map.get(idx)
            else:
                src = ordered[idx] if idx < len(ordered) else None
            if src is None:
                print(f"  [MISSING] {seq}/{idx}.bmp"); missing += 1; continue
            dst = out / "images" / split / seq / f"{idx}.bmp"
            if not dry: place(src, dst, mode)
            rewritten.append(f"{dst.resolve()} {bbox}")

    return rewritten, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="folder with dataN/ subdirs")
    ap.add_argument("--train-ann", required=True, type=Path)
    ap.add_argument("--val-ann", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["copy", "link", "symlink"], default="copy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    for ann, out_name in [(a.train_ann, "train_DAUB.txt"), (a.val_ann, "val_DAUB.txt")]:
        print(f"\nProcessing {ann.name} ...")
        lines, missing = process(ann, a.raw, a.out, a.mode, a.dry_run)
        print(f"  -> {len(lines)} frames placed, {missing} missing")
        if not a.dry_run:
            (a.out / out_name).write_text("\n".join(lines) + "\n")
            print(f"  -> wrote {a.out / out_name}")
    print("\nDone." + ("  (dry run — no files written)" if a.dry_run else ""))

if __name__ == "__main__":
    main()