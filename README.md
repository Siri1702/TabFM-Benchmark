# TabFM Benchmark

A fair, reproducible benchmark comparing **tabular foundation models** (TabPFN) against traditional gradient boosted trees (XGBoost, LightGBM, CatBoost) and neural networks (MLP) on binary classification tasks.

## 📊 Overview

This project investigates the practical trade-offs between **zero-shot tabular foundation models** and **traditionally-tuned** machine learning approaches. The key question:

> Can a single zero-shot model (TabPFN) compete with extensively-tuned gradient boosted trees — without the computational cost of hyperparameter optimization?

### What's Being Compared

| Model | Type | Tuning Required | Notes |
|-------|------|-----------------|-------|
| **TabPFN** | Tabular Foundation Model | ❌ None (zero-shot) | Stores training data as context; no gradient-based training |
| **XGBoost** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Industry-standard gradient boosting |
| **LightGBM** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Fast histogram-based gradient boosting |
| **CatBoost** | Gradient Boosted Trees | ✅ Optuna (50 trials) | Ordered boosting with symmetric trees |
| **MLP** | Neural Network | ✅ Optuna (30 trials) | Scikit-learn MLPClassifier |

### Key Metrics Tracked

- **Performance**: ROC AUC, Average Precision, F1 Macro, Brier Score, Log Loss
- **Timing**: Tuning time, Fit time, Predict time, Total wall time
- **Statistical**: Wilcoxon signed-rank tests for pairwise significance

## 📁 Project Structure

```
tabfm-benchmark/
├── configs/               # Configuration files
│   ├── datasets.yaml     # 15 OpenML datasets
│   ├── models.yaml       # Model configs & hyperparameter search spaces
│   └── experiment.yaml   # Experiment settings
├── src/
│   ├── data/             # Data loading (OpenML)
│   ├── models/           # Model wrappers
│   │   ├── tabpfn_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   └── mlp_model.py
│   ├── evaluation/       # Metrics & statistical tests
│   └── viz/             # Visualizations
├── experiments/
│   ├── run_benchmark.py # Main benchmark runner
│   └── results/         # Raw & aggregated results
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Siri1702/tabfm-benchmark.git
cd tabfm-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Benchmark

```bash
# Run on specific datasets
python experiments/run_benchmark.py --datasets credit-g diabetes

# Run on all 15 datasets
python experiments/run_benchmark.py --all

# Quick test with fewer seeds
python experiments/run_benchmark.py --dataset credit-g --n_seeds 3
```

### Output

Results are saved to:
- `experiments/results/raw/` — Individual JSON results per dataset
- `experiments/results/aggregated/` — Aggregated CSV with all metrics
- `reports/figures/` — Visualizations (heatmaps, rank plots, timing comparisons)

## 📈 Sample Results

From initial experiments on 3 datasets (10 seeds each):

| Model | ROC AUC (mean) | Total Time (s) | Speedup vs XGBoost |
|-------|----------------|----------------|-------------------|
| TabPFN | 0.87 | ~3s | **55×** |
| XGBoost | 0.86 | ~175s | 1.0× |
| LightGBM | 0.85 | ~85s | 2.1× |
| CatBoost | 0.85 | ~245s | 0.7× |
| MLP | 0.84 | ~50s | 3.5× |

> **Key Insight**: TabPFN achieves competitive performance with **zero tuning time**, making it attractive for rapid prototyping and exploration.

## 🧪 Datasets

15 OpenML binary classification datasets, curated for TabPFN's sweet spot (N < 10,000, p < 100):

- Credit-g (German Credit)
- Diabetes (Pima Indians)
- Spambase
- Banknote Authentication
- Hill-Valley
- WDBC (Breast Cancer)
- QSAR-Biodeg
- Titanic
- Churn
- Ozone Level
- KC1 / PC1 (Software Defect Prediction)
- First-Order-Theorem
- Phoneme
- Australian Credit

## 🔬 Methodology

1. **Data Split**: 80% train / 20% test (stratified)
2. **Cross-Validation**: 5-fold stratified for hyperparameter tuning
3. **Seeds**: 10 random seeds per dataset for robust comparison
4. **TabPFN**: Subsamples training data if >3000 rows (practical CPU limit)
5. **Statistical Tests**: Wilcoxon signed-rank for pairwise significance (p < 0.05)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ideas for Contributions

- Add more datasets
- Implement additional models (FT-Transformer, SAINT, NODE)
- Add more statistical tests (Friedman test, Nemenyi post-hoc)
- Improve visualizations
- Add confidence intervals to results
- Support regression tasks

## 📝 License

MIT License — feel free to use this for your own research or projects.

## 📚 Citations

If you use this benchmark in your research, please cite:

```bibtex
@misc{tabfm-benchmark,
  author = {Siri1702},
  title = {TabFM Benchmark: Tabular Foundation Models vs Gradient Boosted Trees},
  year = {2025},
  url = {https://github.com/Siri1702/tabfm-benchmark}
}
```

---

⭐ **Star this repo** if you find it useful for your research!