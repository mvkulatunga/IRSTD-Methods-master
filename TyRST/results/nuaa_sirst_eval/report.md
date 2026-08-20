# TY-RIST Evaluation Report — nuaa_sirst

- **Dataset:** nuaa_sirst
- **Image size:** 640
- **Prune kind:** p2p3
- **Generated:** 2026-06-20 16:24:56

## Results vs Paper

| Metric | Stage1 (no CA) | Stage2 (with CA) | Pruned (p2p3) | Paper | \|% diff (Stage 1 without CA)\| | \|% diff (Stage 2, with CA)\| | \|% diff (Pruned)\| |
|---|---|---|---|---|---|---|---|
| **Params (M)** | 2.78 | 2.78 | 1.75 | 2.10 | 32.2% | 32.4% | 16.8% |
| **GFLOPs** | 50.60 | 50.77 | 44.78 | 40.30 | 25.6% | 26.0% | 11.1% |
| Precision | 95.5% | 95.8% | 95.8% | 92.9% | 2.8% | 3.1% | 3.1% |
| Recall | 94.0% | 89.4% | 89.4% | 92.1% | 2.1% | 3.0% | 3.0% |
| F1 | 94.8% | 92.5% | 92.5% | 92.5% | 2.4% | 0.0% | 0.0% |
| mAP50 | 97.9% | 95.1% | 95.1% | — | — | — | — |

> Params/GFLOPs are fused, measured at the eval resolution (640px). Paper complexity figures: ITSDT/IRDST 2.03M/37.40G @512, NUAA 2.10M/40.30G @640 (Table 1 & 3); NUDT/combined inferred.
