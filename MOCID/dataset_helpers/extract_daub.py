import zipfile
from pathlib import Path
from collections import Counter

SRC = Path("./datasets/DAUB_Zip")              # folder holding the .zip files
DST = Path("./DAUB")    # where to unzip
DST.mkdir(exist_ok=True)

# 1. Unzip all archives
for z in sorted(SRC.glob("*.zip")):
    out = DST / z.stem
    if out.exists():
        continue
    print(f"Extracting {z.name} -> {out}")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out)

# 2. Report structure of each extracted folder
print("\n=== STRUCTURE REPORT ===")
for folder in sorted(DST.iterdir()):
    if not folder.is_dir():
        continue
    exts = Counter(p.suffix.lower() for p in folder.rglob("*") if p.is_file())
    n_files = sum(exts.values())
    subdirs = [d.name for d in folder.iterdir() if d.is_dir()]
    print(f"\n{folder.name}: {n_files} files, exts={dict(exts)}")
    if subdirs:
        print(f"   subdirs: {subdirs[:10]}")
    # show a couple of sample filenames
    sample = [p.name for p in list(folder.rglob('*'))[:5] if p.is_file()]
    print(f"   samples: {sample}")