# TY-RIST Evaluation Report — combined_sirst

- **Dataset:** combined_sirst
- **Image size:** 640
- **Prune kind:** p2p3
- **Generated:** 2026-06-23 10:49:57

## Results vs Paper

| Metric | Stage1 (no CA) | Stage2 (with CA) | Pruned (p2p3) | Paper | \|% diff (Stage 1 without CA)\| | \|% diff (Stage 2, with CA)\| | \|% diff (Pruned)\| |
|---|---|---|---|---|---|---|---|
| **Params (M)** | 2.78 | 2.78 | 1.75 | 2.10 | 32.2% | 32.4% | 16.8% |
| **GFLOPs** | 50.60 | 50.77 | 44.78 | 40.30 | 25.6% | 26.0% | 11.1% |
| Precision | 77.5% | 78.9% | 78.9% | 81.0% | 4.3% | 2.5% | 2.5% |
| Recall | 75.4% | 75.6% | 75.6% | 75.2% | 0.3% | 0.6% | 0.6% |
| F1 | 76.4% | 77.2% | 77.2% | 78.0% | 2.0% | 1.0% | 1.0% |
| mAP50 | 74.1% | 75.3% | 75.3% | — | — | — | — |
