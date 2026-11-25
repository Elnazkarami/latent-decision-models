# Changelog

User-facing changes by release: what you can do with the package, and any correction to a published result. Day-to-day development is in the git history.

## Unreleased

**Fixes:**
- The ds007474 loader ignores hidden files. Before, macOS `._*` metadata files on exFAT/FAT drives were read as behavior files and loading failed.

**Reproducibility:**
- [`reproduce/cleanroom.sh`](reproduce/cleanroom.sh) regenerates every committed report from a fresh clone. It can reuse an earlier, hash-checked OSF download (`OSF_LOCAL`) when the OSF API is unavailable. Results: [validation, part 2](docs/VALIDATION.md).

## 0.1.0 (2026-11-25)

First release.

**What it provides:**
- **Task adapters** for OpenNeuro ds002158 (confidence after one's own and observed decisions; behavior, EEG, fMRI) and ds007474 (two-step task; behavior from OSF). They validate strictly and align events to recordings by timing.
- **Confidence modeling:** behavioral baselines with subject-disjoint evaluation, a second-order signal-detection model with recovery studies, and metacognitive sensitivity.
- **Two-step reinforcement learning:** model-free, model-based and hybrid models, win-stay/lose-shift and arbitration variants, empirical-Bayes fitting, and recovery on real task schedules.
- **In-scanner EEG:** gradient and heartbeat artifact removal (template subtraction, OBS, ICA), a planted-signal retention test, epochs, and leave-one-run-out / leave-one-subject-out decoding with permutation and group tests.
- **fMRI:** metadata audit without downloading images, design diagnostics, budgeted derivative downloads, single-trial (LSS) ROI estimates with optional HRF derivatives, positive controls, and a timing diagnostic.
- **Behavior + EEG fusion:** stacked linear and small PyTorch models.
- **[Research report](docs/REPORT.md):** every result with its evidence level.

**Corrections to reported results:**
- The ds002158 fusion write-up first quoted the earlier "confidence beyond behavior" EEG decoder as r = 0.16. It is r = 0.08 within subject. The fusion conclusion is unchanged.
- Fixes from a code review before this release changed no reported numbers. They covered undefined-score p-values, Holm families with untested cells, and crashes in edge cases.
