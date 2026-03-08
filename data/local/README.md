# Local Data Layout

Place your own train and test files in the folders below.
Do not commit raw datasets.

## Folder layout

- `data/local/train/fdc/`: PatchTST training files
- `data/local/train/vib/`: SwinMAE training files
- `data/local/test/fdc/`: PatchTST test files
- `data/local/test/vib/`: SwinMAE test files

## Supported file formats

### FDC

- Default config uses `*.csv`
- `*.parquet` is also supported if you change `data.source` and `data.path`
- Optional timestamp column: `timestamp` or `time`
- Every non-timestamp column is treated as a numeric feature channel

### Vibration

- Default config uses `*.csv`
- `*.npy` is also supported if you change `data.source` and `data.path`
- CSV columns must include `x`, `y`, `z`
- Optional timestamp column: `timestamp` or `time`

## Default commands

Train everything:

```bash
python run_local_train.py
```

Run test scoring and export JSON/CSV results:

```bash
python run_local_test.py
```

Optional single-stream commands:

```bash
python run_local_train.py --skip-swinmae
python run_local_test.py --stream patchtst
```
