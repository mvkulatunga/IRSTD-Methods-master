# TY-RIST Evaluation Report — itsdt_15k

- **Dataset:** itsdt_15k
- **Image size:** 512
- **Prune kind:** p2
- **Generated:** 2026-06-23 20:07:52

## Results vs Paper

| Metric | Stage1 (no CA) | Stage2 (with CA) | Pruned (p2) | Paper | \|% diff (Stage 1 without CA)\| | \|% diff (Stage 2, with CA)\| | \|% diff (Pruned)\| |
|---|---|---|---|---|---|---|---|
| **Params (M)** | 2.78 | 2.78 | 1.60 | 2.03 | 36.7% | 37.0% | 21.0% |
| **GFLOPs** | 32.39 | 32.49 | 23.96 | 37.40 | 13.4% | 13.1% | 35.9% |
| Precision | 96.7% | 96.4% | 96.1% | — | — | — | — |
| Recall | 94.9% | 95.0% | 95.2% | — | — | — | — |
| F1 | 95.8% | 95.7% | 95.7% | 83.3% | 15.0% | 14.9% | 14.9% |
| mAP50 | 96.6% | 95.9% | 96.0% | 86.8% | 11.3% | 10.5% | 10.5% |
