"""
Main benchmark runner.

Usage:
    python experiments/run_benchmark.py --datasets credit-g diabetes spambase
    python experiments/run_benchmark.py --all           # run all 15 datasets
    python experiments/run_benchmark.py --dataset credit-g --n_seeds 5  # quick test

Results are saved to experiments/results/raw/ as JSON and
aggregated to experiments/results/aggregated/results.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.tabpfn_model import TabPFNWrapper
from src.models.xgboost_model import XGBoostWrapper
from src.models.lightgbm_model import LightGBMWrapper
from src.models.catboost_model import CatBoostWrapper
from src.models.mlp_model import MLPWrapper
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical import wilcoxon_pairwise, average_rank_table
from src.viz.leaderboard import plot_auc_heatmap, plot_average_ranks
from src.viz.leaderboard import plot_timing_comparison


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_models(exp_cfg: dict, seed: int) -> list:
    """Instantiate all model wrappers for a given seed."""
    return [
        TabPFNWrapper(device="cpu", random_state=seed),
        XGBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 50),
                       random_state=seed),
        LightGBMWrapper(n_trials=exp_cfg.get("n_optuna_trials", 50),
                        random_state=seed),
        CatBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 50),
                        random_state=seed),
        MLPWrapper(n_trials=exp_cfg.get("n_optuna_trials", 30),
                   random_state=seed),
    ]


def run_single(dataset_cfg: dict, exp_cfg: dict) -> list[dict]:
    """
    Run all models × all seeds on a single dataset.
    Returns a list of metric dicts (one per model × seed).
    """
    loader = DataLoader(
        test_size=exp_cfg.get("test_size", 0.2),
        random_state=42,  # fixed split seed — the same train/test for ALL model seeds
    )
    dataset = loader.load_openml(
        dataset_id=dataset_cfg["id"],
        dataset_name=dataset_cfg["name"],
    )
    
    # Skip datasets that are too large for TabPFN even with subsampling
    if dataset.n_rows > exp_cfg.get("max_rows", 10000):
        print(f"  Skipping {dataset_cfg['name']}: {dataset.n_rows} rows > limit")
        return []

    seeds = list(range(exp_cfg.get("n_seeds", 10)))
    all_metrics = []

    for seed in tqdm(seeds, desc=f"  Seeds for {dataset_cfg['name']}", leave=False):
        models = build_models(exp_cfg, seed)
        for model in models:
            print(f"    {model.name:12s} | seed={seed}", end=" ")
            result = model.run(
                dataset.X_train, dataset.y_train,
                dataset.X_test,  dataset.y_test,
                dataset_name=dataset_cfg["name"],
                seed=seed,
            )
            metrics = compute_metrics(result)
            all_metrics.append(metrics)

            status = f"AUC={metrics.get('roc_auc', float('nan')):.4f}" \
                     if metrics.get("error") is None \
                     else f"ERROR: {metrics['error']}"
            print(f"→ {status} ({result.fit_time_sec:.1f}s fit)")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Run TabFM benchmark")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=None)
    args = parser.parse_args()

    # Load configs
    dataset_cfgs = load_config("configs/datasets.yaml")["datasets"]
    exp_cfg      = load_config("configs/experiment.yaml")["experiment"]

    if args.n_seeds:
        exp_cfg["n_seeds"] = args.n_seeds
    if args.datasets:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in args.datasets]

    Path("experiments/results/raw").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/aggregated").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    all_results = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_cfg in dataset_cfgs:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_cfg['name']} (OpenML {dataset_cfg['id']})")
        print(f"{'='*60}")
        
        metrics_list = run_single(dataset_cfg, exp_cfg)
        all_results.extend(metrics_list)

        # Save raw results after each dataset (don't lose work if crashes)
        raw_path = f"experiments/results/raw/{run_id}_{dataset_cfg['name']}.json"
        with open(raw_path, "w") as f:
            json.dump(metrics_list, f, indent=2, default=str)

    if not all_results:
        print("No results collected. Check your dataset configs.")
        return

    # Aggregate
    results_df = pd.DataFrame(all_results)
    agg_path = f"experiments/results/aggregated/results_{run_id}.csv"
    results_df.to_csv(agg_path, index=False)
    print(f"\nSaved aggregated results → {agg_path}")

    # Statistical analysis
    print("\n--- Statistical Analysis ---")
    rank_df = average_rank_table(results_df, metric="roc_auc")
    print(rank_df.to_string(index=False))

    wilcoxon_df = wilcoxon_pairwise(results_df, metric="roc_auc")
    print("\nWilcoxon pairwise tests (significant pairs):")
    if not wilcoxon_df.empty:
        print(wilcoxon_df[wilcoxon_df["significant"]].to_string(index=False))

    # Visualisations
    print("\n--- Generating Visualisations ---")
    plot_auc_heatmap(results_df, save_path="reports/figures/auc_heatmap.png")
    plot_average_ranks(rank_df, save_path="reports/figures/average_ranks.png")
    plot_timing_comparison(results_df, save_path="reports/figures/timing_comparison.png")
    print("\n--- Timing Summary (mean across all datasets + seeds) ---")
    timing_summary = (
    results_df
    .groupby("model")[["tuning_time_sec", "fit_time_sec", "predict_time_sec", "total_wall_time_sec"]]
    .mean()
    .round(2)
)
    timing_summary["speedup_vs_xgboost"] = (
    timing_summary.loc["XGBoost", "total_wall_time_sec"] 
    / timing_summary["total_wall_time_sec"]
).round(1)
    print(timing_summary.to_string())

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()