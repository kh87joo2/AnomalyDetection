Use these notebooks for Colab GPU smoke validation:
- `notebooks/colab_patchtst_ssl.ipynb`
- `notebooks/colab_swinmae_ssl.ipynb`

For local execution without notebooks, use:
- `python3 -m pipelines.run_local_training_pipeline --repo-root . --patch-config configs/patchtst_ssl_local.yaml --swin-config configs/swinmae_ssl_local.yaml --patch-data-source csv --patch-data-path "/absolute/path/to/patchtst_data/*.csv" --swin-data-source csv --swin-data-path "/absolute/path/to/swinmae_data/**/*.csv" --persist-run-history --validate-skip-smoke`

For dashboard-driven local training (import files + Train button), run:
- `python3 -m training_dashboard.server --host 127.0.0.1 --port 8765`

For CLI equivalents, see `docs/runbook.md`.
