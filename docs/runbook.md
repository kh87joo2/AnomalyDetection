# Runbook (Phase 1)

## Colab GPU setup
1. Runtime -> Change runtime type -> GPU
2. Install deps: `pip install -r requirements.txt`

## Train commands
- PatchTST: `python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl.yaml`
- SwinMAE: `python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl.yaml`

## Local sanity check
- Compile modules: `python -m compileall core datasets models trainers inference`
- Run smoke tests: `pytest -q`

## Checkpoint verification
- `ls -lh checkpoints/patchtst_ssl.pt checkpoints/swinmae_ssl.pt`
- Confirm both files exist and have non-zero size.

## Loss curve verification
- `ls -lh artifacts/loss/*_loss_history.csv artifacts/loss/*_loss_curve.png`
- Verify train/val loss curves are generated for each stream.
- Optional interactive view: `tensorboard --logdir runs`

## Scoring example command
- PatchTST: `python -m inference.run_scoring_example --stream patchtst --checkpoint checkpoints/patchtst_ssl.pt --config configs/patchtst_ssl.yaml`
- SwinMAE: `python -m inference.run_scoring_example --stream swinmae --checkpoint checkpoints/swinmae_ssl.pt --config configs/swinmae_ssl.yaml`

## Training completion checklist (automated)
- Run: `python -m pipelines.validate_training_outputs`
- Output format:
  - `[v] PASS` or `[ ] FAIL` per checklist item.
  - Summary with passed/failed counts.

## Dashboard state export (Phase 2)
- Generate runtime state JSON:
  - `python -m pipelines.export_training_dashboard_state --repo-root . --out training_dashboard/data/dashboard-state.json`
- Optional smoke-included export:
  - `python -m pipelines.export_training_dashboard_state --repo-root . --out training_dashboard/data/dashboard-state.json --run-smoke`
- Start dashboard static server:
  - `python -m http.server 8765 --directory training_dashboard`

## Local CUDA PC migration
- Copy repo as-is.
- Install compatible CUDA PyTorch build.
- Run the same commands used in Colab.
