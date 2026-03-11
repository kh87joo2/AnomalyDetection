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

## Phase 3A Local GPU migration profile
This stage prepares the same batch decision contracts for a local GPU workstation after Colab validation is complete.

1. Create a fresh venv on the local machine:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python3 -m pip install -r requirements.txt -r requirements-dev.txt`
2. Confirm the validated artifacts exist locally:
   - `checkpoints/patchtst_ssl.pt`
   - `checkpoints/swinmae_ssl.pt`
   - `artifacts/scaler_fdc.json`
   - `artifacts/thresholds/batch_decision_thresholds.json`
3. Place test input files at the local profile paths or edit only the config paths:
   - `runtime_inputs/fdc/test_fdc.csv`
   - `runtime_inputs/vibration/test_vibration.csv`
4. Validate the local GPU profile:
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --dry-run`
5. Run the local batch decision path:
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --run`
6. Expected result:
   - the same report contract as Colab is generated
   - `artifacts/batch_decision/local_gpu_validation/` contains `decision_report.json`, `decision_events.csv`, and `chart_payload.json`
   - `training_dashboard/data/batch-decision-state.json` is refreshed for the dashboard
7. Migration rule:
   - device and path changes stay in config files only; model/scoring/reporting code remains unchanged

## Phase 3A Batch Decision Colab profile
This stage validates the Colab execution profile and runner contract only.
It does not run real batch scoring yet; import/preprocess/scoring arrive in `P0D` and `P0E`.

1. Pull the latest `main` in Colab:
   - `cd /content/AnomalyDetection`
   - `git checkout main`
   - `git pull --ff-only origin main`
2. Install runtime and test dependencies:
   - `python3 -m pip install -r requirements.txt -r requirements-dev.txt`
3. Confirm the profile files exist:
   - `ls -la configs/batch_decision_runtime_colab.yaml`
   - `ls -la artifacts/thresholds/batch_decision_thresholds.json`
4. Validate the Colab profile with the batch runner:
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --dry-run`
5. Run the P0C validation tests:
   - `python3 -m pytest -q tests/batch_decision/test_runner_skeleton.py tests/batch_decision/test_colab_profile.py`
6. Expected result:
   - the dry-run prints `batch_decision dry-run validation passed`
   - pytest passes for the runner skeleton and Colab profile tests

## Phase 3A Batch Import and Preprocess validation
This stage validates test-data import and training-compatible window building.
It still does not run model scoring yet.

1. Pull the latest `main` in Colab:
   - `cd /content/AnomalyDetection`
   - `git checkout main`
   - `git pull --ff-only origin main`
2. Install runtime and test dependencies:
   - `python3 -m pip install -r requirements.txt -r requirements-dev.txt`
3. Run the import/preprocess test target:
   - `python3 -m pytest -q tests/batch_decision/test_import_and_preprocess.py`
4. Expected result:
   - pytest passes for the valid FDC/vibration sample inputs
   - malformed timestamp/axis fixtures are rejected by the preprocessing wrappers

## Phase 3A Batch Scoring validation
This stage validates checkpoint/scaler/config reuse and emits per-window score payloads.
It still does not apply decision thresholds or export reports yet.

1. Pull the latest `main` in Colab:
   - `cd /content/AnomalyDetection`
   - `git checkout main`
   - `git pull --ff-only origin main`
2. Install runtime and test dependencies:
   - `python3 -m pip install -r requirements.txt -r requirements-dev.txt`
3. Run the scoring tests:
   - `python3 -m pytest -q tests/batch_decision/test_scoring_engine.py tests/batch_decision/test_runner_score_only.py`
4. Before the real CLI run, confirm `configs/batch_decision_runtime_colab.yaml` points to real test files that match your trained artifacts:
   - `run.input_paths.patchtst`
   - `run.input_paths.swinmae`
5. Run the score-only CLI with real artifacts:
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --score-only`
6. Expected result:
   - pytest passes for the scoring engine and score-only runner tests
   - the CLI prints `batch_decision score-only run completed`
   - stream window counts and sample score values are printed for the configured inputs

## Phase 3A Decision and Reporting validation
This stage validates threshold-based decisions, reason fields, and JSON/CSV/chart exports.

1. Pull the latest `main` in Colab:
   - `cd /content/AnomalyDetection`
   - `git checkout main`
   - `git pull --ff-only origin main`
2. Install runtime and test dependencies:
   - `python3 -m pip install -r requirements.txt -r requirements-dev.txt`
3. Run the decision/reporting tests:
   - `python3 -m pytest -q tests/batch_decision/test_decision_engine.py tests/batch_decision/test_reporting.py tests/batch_decision/test_runner_full_run.py tests/batch_decision/test_runner_skeleton.py`
4. Before the full CLI run, confirm `configs/batch_decision_runtime_colab.yaml` points to real test files that match your trained artifacts:
   - `run.input_paths.patchtst`
   - `run.input_paths.swinmae`
5. Run the full batch decision path:
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --run`
6. Expected result:
   - pytest passes for decision/reporting and full runner tests
   - the CLI prints `batch_decision run completed`
   - `decision_counts` is printed
   - output directory contains `decision_report.json`, `decision_events.csv`, and `chart_payload.json`
   - dashboard bridge file is refreshed at `training_dashboard/data/batch-decision-state.json`

## Phase 3A Dashboard bridge
After a successful `--run`, the dashboard can visualize batch decision output in a separate tab.

1. Confirm both runtime files exist:
   - `training_dashboard/data/dashboard-state.json`
   - `training_dashboard/data/batch-decision-state.json`
2. Start the static dashboard server:
   - `python -m http.server 8765 --directory training_dashboard`
3. Open `http://127.0.0.1:8765`
4. Use:
   - `Training Flow` / `Artifact Gate` tabs for training artifacts
   - `Batch Decision` tab for imported test-data scoring and threshold overlays
