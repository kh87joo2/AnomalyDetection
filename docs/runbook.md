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

## Scoring example command
- PatchTST: `python -m inference.run_scoring_example --stream patchtst --checkpoint checkpoints/patchtst_ssl.pt --config configs/patchtst_ssl.yaml`
- SwinMAE: `python -m inference.run_scoring_example --stream swinmae --checkpoint checkpoints/swinmae_ssl.pt --config configs/swinmae_ssl.yaml`

## Local CUDA PC migration
- Copy repo as-is.
- Install compatible CUDA PyTorch build.
- Run the same commands used in Colab.
