# TY-RIST Evaluation Report — nudt_sirst

- **Dataset:** nudt_sirst
- **Image size:** 640
- **Prune kind:** p2
- **Generated:** 2026-06-20 16:21:06

## Results vs Paper

| Metric | Stage1 (no CA) | Stage2 (with CA) | Pruned (p2) | Paper | \|% diff (Stage 1 without CA)\| | \|% diff (Stage 2, with CA)\| | \|% diff (Pruned)\| |
|---|---|---|---|---|---|---|---|
| **Params (M)** | 2.78 | 2.78 | 1.60 | 2.03 | 36.7% | 37.0% | 21.0% |
| **GFLOPs** | 50.60 | 50.77 | 37.43 | — | — | — | — |
| Precision | 99.7% | 99.5% | 94.7% | 96.8% | 3.0% | 2.8% | 2.1% |
| Recall | 95.1% | 94.1% | 75.8% | 95.8% | 0.7% | 1.7% | 20.8% |
| F1 | 97.4% | 96.7% | 84.2% | 96.3% | 1.1% | 0.4% | 12.5% |
| mAP50 | 98.0% | 96.3% | 87.1% | — | — | — | — |

> Params/GFLOPs are fused, measured at the eval resolution (640px). Paper complexity figures: ITSDT/IRDST 2.03M/37.40G @512, NUAA 2.10M/40.30G @640 (Table 1 & 3); NUDT/combined inferred.
