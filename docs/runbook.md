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

## Interactive dashboard training (upload + Train button)
Use the dashboard API server (not plain `http.server`) when you want to import files and trigger training from UI:

```bash
python3 -m training_dashboard.server --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765` and:
1. Import PatchTST files (CSV/Parquet or ZIP containing those).
2. Import SwinMAE files (CSV/NPY or ZIP containing those).
3. Click `Train`.
4. Watch node glow state (`running/done/fail`) to track the active step in real time.

The dashboard runner executes:
- `pipelines.run_local_training_pipeline`
- scoring for both streams
- validation checklist
- dashboard-state export with run history persistence

## Notebook-free local workflow (recommended)
Use the integrated Python workflow to replace notebook execution end-to-end.
This workflow does **not** download Kaggle datasets. Provide local dataset paths directly.

```bash
python3 -m pipelines.run_local_training_pipeline \
  --repo-root . \
  --patch-config configs/patchtst_ssl_local.yaml \
  --swin-config configs/swinmae_ssl_local.yaml \
  --patch-data-source csv \
  --patch-data-path "/absolute/path/to/patchtst_data/*.csv" \
  --swin-data-source csv \
  --swin-data-path "/absolute/path/to/swinmae_data/**/*.csv" \
  --persist-run-history \
  --validate-skip-smoke
```

Useful options:
- Dry run command preview only: `--dry-run`
- Skip one stream: `--skip-patchtst` or `--skip-swinmae`
- Skip full steps: `--skip-scoring`, `--skip-validate`, `--skip-export`
- Force smoke validation inside export: `--run-smoke`
- Runtime config output location: `--runtime-config-dir artifacts/runtime_configs`

## Local CUDA PC migration
- Copy repo as-is.
- Install compatible CUDA PyTorch build.
- Run the same commands used in Colab, or use the notebook-free local workflow above.
