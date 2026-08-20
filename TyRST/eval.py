import argparse
import csv
from pathlib import Path
from datetime import datetime

import ultralytics.nn.tasks
from ultralytics import YOLO
import components                                   # so CoordAtt unpickles
ultralytics.nn.tasks.CoordAtt = components.CoordAtt
from ultralytics.nn.tasks import DetectionModel
DetectionModel.init_criterion = lambda self: components.TYRISTDetectionLoss(self)
from ultralytics.utils.torch_utils import get_num_params, get_flops

PROJECT = "TY-RIST_Project"
MULTIFRAME_TAGS = {"itsdt_15k", "irdst"}            # eval at 512, else 640 (Sec 4.3)

PRUNE_TARGET = {
    "nuaa_sirst": "p2p3", "nudt_sirst": "p2", "combined_sirst": "p2p3",
    "itsdt_15k": "p2",    "irdst": "p2",
}

# Paper benchmark numbers. Metric keys (%) the paper reports per dataset, plus
# params (millions) and gflops where the paper gives them (Table 1 & Table 3).
#   ITSDT-15k : Table 3 upper last row -> 2.03M / 37.40G (512, P2-only -PAN)
#   NUAA-SIRST: Table 3 lower last row -> 2.10M / 40.30G (640, P2+P3 partial-PAN)
#   IRDST     : Table 1 multi-frame row -> 2.03M / 37.40G  (P2-only, same as ITSDT) [inferred]
#   NUDT-SIRST: P2-only like ITSDT -> ~2.03M params; FLOPS not reported at 640 -> None [inferred]
#   combined  : cross-dataset on IRDST-1k, NUAA-style P2+P3 -> 2.10M / 40.30G [inferred]
PAPER = {
    "nuaa_sirst":     {"precision": 92.9, "recall": 92.1, "f1": 92.5, "params": 2.10, "gflops": 40.30},
    "nudt_sirst":     {"precision": 96.8, "recall": 95.8, "f1": 96.3, "params": 2.03, "gflops": None},
    "combined_sirst": {"precision": 81.0, "recall": 75.2, "f1": 78.0, "params": 2.10, "gflops": 40.30},
    "itsdt_15k":      {"map50": 86.80, "f1": 83.26, "params": 2.03, "gflops": 37.40},
    "irdst":          {"map50": 89.90, "f1": 90.40, "params": 2.03, "gflops": 37.40},
}
LABEL = {"precision": "Precision", "recall": "Recall", "f1": "F1", "map50": "mAP50",
         "params": "Params (M)", "gflops": "GFLOPs"}
METRIC_ORDER = ["precision", "recall", "f1", "map50"]      # accuracy metrics
COMPLEXITY_ORDER = ["params", "gflops"]                    # efficiency metrics

STAGE_DIR = {"Stage1 (no CA)": "val-stage-1", "Stage2 (with CA)": "val-stage-2"}


def stage_dirname(label):
    return STAGE_DIR.get(label, "pruned")


def paper_keys(paper):
    """Accuracy metrics the paper reports for this dataset (params/gflops handled separately)."""
    return [k for k in METRIC_ORDER if k in paper]


def absdiff(val, ref):
    return abs((val - ref) / ref * 100.0)


def model_complexity(ckpt, imgsz):
    """Return (param_count, GFLOPs) for the fused model at the eval resolution."""
    y = YOLO(ckpt)
    try:
        y.model.float().fuse()                      # match the 'fused' summary numbers
    except Exception:
        pass
    try:
        params = get_num_params(y.model)
    except Exception:
        params = None
    try:
        gflops = round(float(get_flops(y.model, imgsz)), 2)
    except Exception:
        gflops = None
    return params, gflops


def evaluate(ckpt, data_cfg, imgsz, project, name):
    m = YOLO(ckpt).val(data=data_cfg, imgsz=imgsz, split="val", verbose=False,
                       project=project, name=name, exist_ok=True)
    P, R = m.box.mp, m.box.mr
    f1 = 2 * P * R / (P + R + 1e-9)
    return {"precision": P * 100, "recall": R * 100, "f1": f1 * 100, "map50": m.box.map50 * 100}


def report(title, ours, paper):
    print(f"\n{'='*64}\n{title}\n{'='*64}")
    print(f"{'Metric':<12}{'Test-Bench':>12}{'Paper':>10}{'|% diff|':>12}")
    print("-" * 46)
    # complexity rows (params in M to match the paper's units)
    params_m = ours["params"] / 1e6 if ours["params"] is not None else None
    for key, tb in (("params", params_m), ("gflops", ours["gflops"])):
        ref = paper.get(key)
        tb_s = f"{tb:.2f}" if tb is not None else "n/a"
        if ref is not None and tb is not None:
            print(f"{LABEL[key]:<12}{tb_s:>12}{ref:>10.2f}{absdiff(tb, ref):>11.1f}%")
        else:
            print(f"{LABEL[key]:<12}{tb_s:>12}{'—':>10}{'—':>12}")
    # accuracy rows
    for key in paper_keys(paper):
        val, ref = ours[key], paper[key]
        print(f"{LABEL[key]:<12}{val:>11.1f}%{ref:>9.1f}%{absdiff(val, ref):>11.1f}%")
    extra = "  ".join(f"{LABEL[k]} {ours[k]:.1f}" for k in METRIC_ORDER)
    print(f"  [all computed]  {extra}")


def write_csv(path, results, paper):
    cols = (["stage", "params", "gflops"] + METRIC_ORDER
            + ["abs_pct_diff_params", "abs_pct_diff_gflops"]
            + [f"abs_pct_diff_{m}" for m in METRIC_ORDER])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for label, r in results.items():
            params_m = r["params"] / 1e6 if r["params"] is not None else None
            row = [label, r["params"], r["gflops"]] + [round(r[m], 2) for m in METRIC_ORDER]
            # complexity diffs
            row.append(round(absdiff(params_m, paper["params"]), 2)
                       if paper.get("params") and params_m is not None else "")
            row.append(round(absdiff(r["gflops"], paper["gflops"]), 2)
                       if paper.get("gflops") and r["gflops"] is not None else "")
            # accuracy diffs
            for m in METRIC_ORDER:
                row.append(round(absdiff(r[m], paper[m]), 2) if m in paper else "")
            w.writerow(row)


def write_md(path, tag, imgsz, kind, results, paper):
    stages = [s for s in ("Stage1 (no CA)", "Stage2 (with CA)", f"Pruned ({kind})")
              if s in results]
    diff_hdr = {"Stage1 (no CA)": "Stage 1 without CA",
                "Stage2 (with CA)": "Stage 2, with CA",
                f"Pruned ({kind})": "Pruned"}

    L = []
    L.append(f"# TY-RIST Evaluation Report — {tag}\n")
    L.append(f"- **Dataset:** {tag}")
    L.append(f"- **Image size:** {imgsz}")
    L.append(f"- **Prune kind:** {kind}")
    L.append(f"- **Generated:** {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # one wide table: rows = metrics, cols = stages + Paper + |% diff| per stage
    head = (["Metric"] + stages + ["Paper"]
            + [f"\\|% diff ({diff_hdr[s]})\\|" for s in stages])
    L.append("## Results vs Paper\n")
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))

    # Params (M) row
    pref = paper.get("params")
    row = ["**Params (M)**"]
    for s in stages:
        pm = results[s]["params"] / 1e6 if results[s]["params"] is not None else None
        row.append(f"{pm:.2f}" if pm is not None else "n/a")
    row.append(f"{pref:.2f}" if pref is not None else "—")
    for s in stages:
        pm = results[s]["params"] / 1e6 if results[s]["params"] is not None else None
        row.append(f"{absdiff(pm, pref):.1f}%" if (pref and pm is not None) else "—")
    L.append("| " + " | ".join(row) + " |")

    # GFLOPs row
    gref = paper.get("gflops")
    row = ["**GFLOPs**"]
    for s in stages:
        g = results[s]["gflops"]
        row.append(f"{g:.2f}" if g is not None else "n/a")
    row.append(f"{gref:.2f}" if gref is not None else "—")
    for s in stages:
        g = results[s]["gflops"]
        row.append(f"{absdiff(g, gref):.1f}%" if (gref and g is not None) else "—")
    L.append("| " + " | ".join(row) + " |")

    # accuracy metric rows
    for m in METRIC_ORDER:
        row = [LABEL[m]]
        row += [f"{results[s][m]:.1f}%" for s in stages]
        row.append(f"{paper[m]:.1f}%" if m in paper else "—")
        for s in stages:
            row.append(f"{absdiff(results[s][m], paper[m]):.1f}%" if m in paper else "—")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    path.write_text("\n".join(L))


def summary(tag, results, paper):
    print(f"\n{'#'*68}\n# SUMMARY — {tag}  (all stages)\n{'#'*68}")
    stages = [s for s in ("Stage1 (no CA)", "Stage2 (with CA)", f"Pruned ({PRUNE_TARGET[tag]})")
              if s in results]
    header = f"{'':<12}" + "".join(f"{s:>18}" for s in stages) + f"{'Paper':>10}"
    print(header); print("-" * len(header))
    pref = paper.get("params"); gref = paper.get("gflops")
    print(f"{'Params (M)':<12}"
          + "".join(f"{(results[s]['params']/1e6):>18.2f}" for s in stages)
          + (f"{pref:>10.2f}" if pref else f"{'—':>10}"))
    print(f"{'GFLOPs':<12}"
          + "".join(f"{results[s]['gflops']:>18}" for s in stages)
          + (f"{gref:>10.2f}" if gref else f"{'—':>10}"))
    for key in paper_keys(paper):
        row = f"{LABEL[key]:<12}" + "".join(f"{results[s][key]:>17.1f}%" for s in stages)
        row += f"{paper[key]:>9.1f}%"
        print(row)
    if "map50" not in paper:
        print(f"{LABEL['map50']:<12}"
              + "".join(f"{results[s]['map50']:>17.1f}%" for s in stages) + f"{'N/A':>10}")


def main():
    ap = argparse.ArgumentParser(
        description="TY-RIST eval across 3 stages vs paper (metrics + Params/GFLOPs); "
                    "saves report.md + metrics.csv under results/<tag>_eval/.")
    ap.add_argument("--data", required=True, help="e.g. configs/data/nuaa_sirst.yaml")
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--pruned", default=None)
    ap.add_argument("--imgsz", type=int, default=None, help="override eval image size")
    ap.add_argument("--results-dir", default="results", help="root for eval outputs")
    a = ap.parse_args()

    tag = Path(a.data).stem
    if tag not in PAPER:
        raise ValueError(f"No paper reference for '{tag}'. Known: {list(PAPER)}")
    imgsz = a.imgsz or (512 if tag in MULTIFRAME_TAGS else 640)
    kind = PRUNE_TARGET[tag]

    stage1_ckpt = a.stage1 or f"runs/detect/{PROJECT}/Stage1_{tag}/weights/best.pt"
    stage2_ckpt = a.stage2 or f"runs/detect/{PROJECT}/Stage2_{tag}/weights/best.pt"
    pruned_ckpt = a.pruned or f"weights/tyrist_{tag}_{kind}.pt"

    # Everything under ./results/<tag>_eval/ (absolute => val won't prepend runs/detect/).
    eval_root = (Path(a.results_dir) / f"{tag}_eval").resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    val_project = str(eval_root)

    print(f"Dataset: {tag}  |  imgsz: {imgsz}  |  prune kind: {kind}")
    print(f"Saving everything under: {eval_root}/")

    stage_ckpts = (
        ("Stage1 (no CA)",   stage1_ckpt),
        ("Stage2 (with CA)", stage2_ckpt),
        (f"Pruned ({kind})", pruned_ckpt),
    )

    results = {}
    for label, ckpt in stage_ckpts:
        if not Path(ckpt).is_file():
            print(f"\n!! skipping {label}: not found -> {ckpt}")
            continue
        name = stage_dirname(label)
        params, gflops = model_complexity(ckpt, imgsz)
        print(f"\n>>> Evaluating {label}: {ckpt}")
        print(f"    params={params:,}  gflops={gflops}  ->  {eval_root.name}/{name}")
        ours = evaluate(ckpt, a.data, imgsz, val_project, name)
        ours["params"], ours["gflops"] = params, gflops
        results[label] = ours
        report(f"{tag}  —  {label}  vs Paper", ours, PAPER[tag])

    if results:
        summary(tag, results, PAPER[tag])
        csv_path = eval_root / "metrics.csv"
        md_path = eval_root / "report.md"
        write_csv(csv_path, results, PAPER[tag])
        write_md(md_path, tag, imgsz, kind, results, PAPER[tag])
        print(f"\n✓ Saved: {csv_path}")
        print(f"✓ Saved: {md_path}")


if __name__ == "__main__":
    main()