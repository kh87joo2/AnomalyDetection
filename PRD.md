# PRD (Product Requirements Doc) - Phase 1 (MVP Core Only)

## 0) Decision Lock (Applied)

- Code root is current repository folder.
- Training objective is fixed to masked reconstruction for both streams.
- Validation strategy is Google Colab with CUDA GPU; deployment target is local CUDA PC.

## 1) Product Goal

Build a skeleton framework that can independently train and validate two self-supervised anomaly modeling cores with minimal anomaly labels:

- PatchTST-SSL for multivariate FDC time series
- SwinMAE-SSL for vibration (x, y, z) transformed into CWT scalograms

The framework must run in Google Colab now (GPU optional) and be portable to a local CUDA GPU PC later without code redesign.

## 2) In-Scope Features (Phase 1)

### A. PatchTST-SSL (FDC / Time Stream)

- Input: continuous multivariate series `(T, C)` where `C` can be `50+`
- Pipeline: sliding windows -> patching -> random masking -> masked reconstruction training
- Normalization: train-only channel statistics (no leakage)
- Output: trained checkpoint and optional reconstruction-error anomaly score hook

### B. SwinMAE-SSL (Vibration / Physics Stream)

- Input: continuous vibration series `(T, 3)` at configurable `fs` (default 2000 Hz)
- Windowing: default `2 sec` windows with configurable stride
- Transform: Morlet CWT per axis -> stack as 3-channel image -> resize (default `224`)
- Training: masked reconstruction with MSE (masked area focused)
- Output: trained checkpoint and optional reconstruction-error anomaly score hook

## 3) Out of Scope (Phase 1)

- No-code UI, backend API, DB, dashboard/PDF, deployment stack
- Full DQVL, fusion/threshold governance, XAI/correlation, LLM report generation

## 4) User Scenarios

- In Colab, run 1-2 epochs on synthetic data to validate shape/forward/backward/save-load.
- Replace synthetic dataset with real CSV data later without rewriting training loops.
- Run the same training entry points on local CUDA workstation.

## 5) Success Criteria

- End-to-end execution works on Colab CPU and auto-uses CUDA when available.
- Both pipelines complete data loading, preprocessing, training, checkpoint save/load.
- Loss shows a decreasing trend on synthetic smoke tests.
- Config-only changes support FDC channel count/window/mask ratio.
- Config-only changes support vibration `fs`, window, CWT params, frequency range, and image size.

## 6) References

- PatchTST: https://github.com/yuqinie98/PatchTST
- PatchTST paper: https://arxiv.org/abs/2211.14730
- Swin-MAE paper: https://arxiv.org/abs/2212.13805
- Swin-MAE code: https://github.com/Zian-Xu/Swin-MAE
