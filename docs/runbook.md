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

## Phase 3A Local retrain-first workflow
Use this flow when the local GPU workstation will generate fresh training artifacts instead of copying Colab outputs.

1. Create local training config copies:
   - `cp configs/patchtst_ssl.yaml configs/patchtst_ssl_local_train.yaml`
   - `cp configs/swinmae_ssl.yaml configs/swinmae_ssl_local_train.yaml`
2. Edit `configs/patchtst_ssl_local_train.yaml` for real FDC training data:

```yaml
device:
  prefer_cuda: true
  amp: true

logging:
  log_dir: runs/patchtst_ssl
  checkpoint_path: checkpoints/patchtst_ssl.pt

data:
  source: csv
  path: /your/local/train/fdc/*.csv
  timestamp_col: timestamp
  seq_len: 512
  seq_stride: 256
  normalization: robust
```

3. Edit `configs/swinmae_ssl_local_train.yaml` for real vibration training data:

```yaml
device:
  prefer_cuda: true
  amp: true

logging:
  log_dir: runs/swinmae_ssl
  checkpoint_path: checkpoints/swinmae_ssl.pt

data:
  source: csv
  path: /your/local/train/vibration/*.csv
  timestamp_col: timestamp
  fs: 2000
  win_sec: 2.0
  win_stride_sec: 1.0
```

4. Run local training:
   - `python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl_local_train.yaml`
   - `python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl_local_train.yaml`
5. Validate local training outputs:
   - `python -m pipelines.validate_training_outputs --repo-root . --patch-checkpoint checkpoints/patchtst_ssl.pt --swin-checkpoint checkpoints/swinmae_ssl.pt --scaler-path artifacts/scaler_fdc.json --runs-dir runs --patch-config configs/patchtst_ssl_local_train.yaml --swin-config configs/swinmae_ssl_local_train.yaml`
6. Update the batch runtime profile to reuse those exact training-time preprocess configs:

```yaml
run:
  input_paths:
    patchtst: runtime_inputs/fdc/test_fdc.csv
    swinmae: runtime_inputs/vibration/test_vibration.csv

preprocess:
  patchtst_config: configs/patchtst_ssl_local_train.yaml
  swinmae_config: configs/swinmae_ssl_local_train.yaml
```

7. Place test data in:
   - `runtime_inputs/fdc/test_fdc.csv`
   - `runtime_inputs/vibration/test_vibration.csv`
8. Run local batch inference:
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --dry-run`
   - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --run`
9. Optional dashboard preview:
   - `python -m http.server 8765 --directory training_dashboard`

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

## 로컬 재학습 후 테스트 실행 순서 (한글 안내)

큰 순서

1. 코드 받기
2. 학습용 config 2개 만들기
3. PatchTST 학습
4. SwinMAE 학습
5. 학습 산출물 검증
6. 테스트용 config 경로 맞추기
7. test CSV 2개 넣기
8. dry-run
9. `--run`
10. 대시보드 확인

중요한 점 하나:

- 현재 기본 학습 config인 `patchtst_ssl.yaml`, `swinmae_ssl.yaml` 는 기본값이 `synthetic` 입니다.
- 그래서 로컬에서 실데이터로 다시 train 하려면 학습용 config를 따로 하나씩 복사해서 수정하는 방식이 가장 안전합니다.

아래 순서대로 하면 됩니다.

### 1. 최신 코드 받기

```bash
cd /path/to/AnomalyDetection
git checkout main
git pull --ff-only origin main
```

### 2. GPU 확인

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True` 나오면 준비된 상태입니다.

### 3. 의존성 설치

이미 머신러닝 환경이 있다 해도 프로젝트 의존성은 맞춰주는 게 좋습니다.

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 4. 학습용 config 복사

원본을 직접 수정하지 말고 복사해서 쓰는 걸 권장합니다.

```bash
cp configs/patchtst_ssl.yaml configs/patchtst_ssl_local_train.yaml
cp configs/swinmae_ssl.yaml configs/swinmae_ssl_local_train.yaml
```

### 5. PatchTST 학습 config 수정

수정 파일:

- `patchtst_ssl_local_train.yaml`

최소 수정 항목:

- `data.source: csv` 또는 실제 형식
- `data.path: <FDC 학습 데이터 경로>`
- 필요하면 `device.amp`, `training.batch_size`, `training.max_train_batches`, `training.max_val_batches`

핵심은 이 부분입니다.

```yaml
data:
  source: csv
  path: /your/local/train/fdc/*.csv
  timestamp_col: timestamp
```

### 6. SwinMAE 학습 config 수정

수정 파일:

- `swinmae_ssl_local_train.yaml`

최소 수정 항목:

- `data.source: csv` 또는 실제 형식
- `data.path: <vibration 학습 데이터 경로>`

예:

```yaml
data:
  source: csv
  path: /your/local/train/vibration/*.csv
  timestamp_col: timestamp
```

### 7. PatchTST 학습 실행

```bash
python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl_local_train.yaml
```

산출물:

- `checkpoints/patchtst_ssl.pt`
- `artifacts/scaler_fdc.json`
- `runs/patchtst_ssl/...`
- `artifacts/loss/patchtst_loss_history.csv`

### 8. SwinMAE 학습 실행

```bash
python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl_local_train.yaml
```

산출물:

- `checkpoints/swinmae_ssl.pt`
- `runs/swinmae_ssl/...`
- `artifacts/loss/swinmae_loss_history.csv`

### 9. 학습 산출물 검증

이 단계는 꼭 하는 게 좋습니다.

```bash
python -m pipelines.validate_training_outputs \
  --repo-root . \
  --patch-checkpoint checkpoints/patchtst_ssl.pt \
  --swin-checkpoint checkpoints/swinmae_ssl.pt \
  --scaler-path artifacts/scaler_fdc.json \
  --runs-dir runs \
  --patch-config configs/patchtst_ssl_local_train.yaml \
  --swin-config configs/swinmae_ssl_local_train.yaml
```

통과 기준:

- 마지막에 `passed` 요약이 나와야 합니다.

### 10. 테스트용 runtime config 수정

파일:

- `batch_decision_runtime_local_gpu.yaml`

여기서 수정할 것은 2가지입니다.

1. 테스트 CSV 경로

```yaml
run:
  input_paths:
    patchtst: runtime_inputs/fdc/test_fdc.csv
    swinmae: runtime_inputs/vibration/test_vibration.csv
```

2. `preprocess` config 경로  
   이 부분이 중요합니다.  
   방금 학습에 쓴 local train config를 가리키게 바꾸는 게 맞습니다.

```yaml
preprocess:
  patchtst_config: configs/patchtst_ssl_local_train.yaml
  swinmae_config: configs/swinmae_ssl_local_train.yaml
```

이유:

- 배치 테스트는 여기서 학습 때와 같은 전처리/윈도우 조건을 재사용하기 때문입니다.

### 11. 테스트 CSV 준비

폴더 생성:

```bash
mkdir -p runtime_inputs/fdc runtime_inputs/vibration
```

파일 배치:

- FDC 테스트 CSV -> `runtime_inputs/fdc/test_fdc.csv`
- vibration 테스트 CSV -> `runtime_inputs/vibration/test_vibration.csv`

형식:

- FDC: `timestamp` + 수치 컬럼들
- vibration: `x,y,z` 또는 `X-axis/Y-axis/Z-axis` 계열도 현재는 허용

### 12. dry-run 실행

```bash
python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --dry-run
```

이 단계에서 확인하는 것:

- 두 입력 파일 경로 존재
- threshold 파일 존재
- `dual` 모드 계약 정상

### 13. 실제 test 실행

```bash
python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --run
```

정상 결과:

- `batch_decision run completed`
- `decision_counts=...`

생성 파일:

- `artifacts/batch_decision/local_gpu_validation/decision_report.json`
- `artifacts/batch_decision/local_gpu_validation/decision_events.csv`
- `artifacts/batch_decision/local_gpu_validation/chart_payload.json`
- `training_dashboard/data/batch-decision-state.json`

### 14. 대시보드 보기

```bash
python -m http.server 8765 --directory training_dashboard
```

브라우저:

```text
http://127.0.0.1:8765
```

그 다음 `Batch Decision` 탭 클릭

### 실전용으로 더 짧게 요약하면

1. `git pull`
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. 학습 config 2개 복사 후 실데이터 경로 수정
4. `python -m trainers.train_patchtst_ssl --config ...`
5. `python -m trainers.train_swinmae_ssl --config ...`
6. `python -m pipelines.validate_training_outputs ...`
7. `batch_decision_runtime_local_gpu.yaml`에서  
   - test CSV 경로 수정  
   - `preprocess` config 경로를 local train config로 수정
8. `--dry-run`
9. `--run`
10. 대시보드 확인
