# Balance Forgetting and Remembering (BFR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c)](https://pytorch.org/)

Official implementation for the paper:

**Balance Forgetting and Remembering: An Extension of Machine Unlearning for Policy Updates in Machine Learning-Based Access Control**

BFR targets *real-time* policy updates in Machine Learning-Based Access Control (MLBAC) under the stability–plasticity dilemma:
- **Forgetting**: selectively remove outdated policy effects.
- **Remembering**: quickly adapt to new policies while preserving fidelity.

---

## Repository layout

```text
BFR-Policy-Update/
├── src/                      # Core implementation
│   ├── base.py               # Base models (LR / DNN)
│   ├── unlearners.py         # Baselines + BFR + SISA
│   └── puts_factory.py       # Policy Update Task (PUT) generators
├── experiments/              # Reproducible experiment entry points
│   ├── run_main.py           # Main evaluation (AFT curves + runtime)
│   ├── run_ablation.py       # Ablation study (w/o Forget / w/o Remember / Full)
│   └── tune_params.py        # Grid search for BFR hyperparameters
├── data/
│   ├── raw/                  # Place original datasets here (not included)
│   └── processed/            # Cleaned/processed datasets (generated)
├── results/                  # CSV outputs (generated)
└── requirements.txt
```

> **Note**: The repository does not ship the datasets due to licensing/size constraints.

---

## Getting started

### 1) Environment

We recommend using a clean virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Datasets

Download and place datasets under `data/raw/`:

- **AMZN-K (Kaggle)**: Amazon Employee Access Challenge  
- **AMZN-U (UCI)**: Amazon Access Samples

Suggested filenames (you can change them, but keep scripts consistent):
```text
data/raw/train.csv                 # Kaggle AMZN-K
data/raw/uci-2.0.csv               # UCI AMZN-U
```

### 3) Data cleaning (conflict removal)

Real-world access logs may contain **conflicts** (identical attributes with different permissions). Use the provided preprocessing script(s) to generate cleaned data in `data/processed/`.

Example:
```bash
python data/preprocess/check.py \
  --input_file data/raw/uci-2.0.csv \
  --output_file data/processed/cleaned_uci.csv
```

---

## Reproducing experiments

All experiments are runnable from the project root.

### A) Main evaluation (AFT curves)

Runs BFR and baselines, producing AUC / adaptation accuracy / runtime metrics for AFT analysis.

```bash
python experiments/run_main.py
```

**Outputs**
- One or more CSV files under `results/` (e.g., `performance_<dataset>_<scenario>_<base>_<K>.csv`)
- Console logs showing device, settings, and progress.

### B) Ablation study

Compares:
- **Full BFR**
- **w/o Forgetting**
- **w/o Remembering**

```bash
python experiments/run_ablation.py
```

**Outputs**
- Printed summary table
- CSV saved to `results/` (depending on your script settings)

### C) Hyperparameter tuning (grid search)

Searches \(\alpha\) (forgetting rate) and \(\beta\) (retuning rate) by minimizing the selected trade-off objective.

```bash
python experiments/tune_params.py --model LR --dataset kaggle
```

---

## Reproducibility checklist

To help reviewers reproduce results reliably:

- [ ] Use the same Python + package versions (`requirements.txt`)
- [ ] Run with a fixed random seed (supported in code via `set_seed(seed)`)
- [ ] Use the same dataset split (`train_test_split(..., random_state=40)`)
- [ ] Record device info (CPU/GPU) and PyTorch version
- [ ] Keep `results/` intact (raw CSVs are the source of plotted figures)

If you publish artifacts, we recommend uploading:
- `results/*.csv`
- exact command lines used (or shell script)
- `pip freeze > environment.lock.txt`

---

## Configuration knobs (common)

Most scripts expose these variables/args (names may differ slightly across files):

- `MODIFICATION_TYPE`: `mod_labels_only`, `mod_features_only`, `mod_features_and_labels`
- `MODEL_ARCHITECTURE`: `LR` or `DNN`
- `SAMPLE_NUM`: number of PUT items \(K\)
- `NUM_RUNS`: number of PUT scenarios \(J\)
- `NUM_POINTS`: grid resolution for tuning/curve generation
- `N_SHARDS`: shards for SISA

---

## Troubleshooting

- **CUDA not used**: Ensure your PyTorch build matches your CUDA runtime; otherwise it will fall back to CPU.
- **Dataset path errors**: Verify files exist under `data/raw/` or update script paths accordingly.
- **Memory issues**: One-hot encoding may expand dimensions; reduce batch size or run on CPU for small-scale tests.

---

## License

This project is released under the MIT License. See `LICENSE`.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Anonymous2025BFR,
  title={Balance Forgetting and Remembering: An Extension of Machine Unlearning for Policy Updates in Machine Learning-Based Access Control},
  author={Anonymous Author(s)},
  journal={Under Review},
  year={2025}
}
```

---

## Contact

For questions or issues, please open a GitHub issue in this repository.
