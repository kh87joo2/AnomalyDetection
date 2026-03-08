Use these notebooks for Colab GPU smoke validation:
- `notebooks/colab_patchtst_ssl.ipynb`
- `notebooks/colab_swinmae_ssl.ipynb`

For CLI equivalents, see `docs/runbook.md`.

Phase 3A batch decision profile validation is CLI-first:
- `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --dry-run`
- `python3 -m pytest -q tests/batch_decision/test_runner_skeleton.py tests/batch_decision/test_colab_profile.py`
