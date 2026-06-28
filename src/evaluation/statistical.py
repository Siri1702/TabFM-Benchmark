"""
Statistical significance testing across multiple datasets and models.
Implements the Wilcoxon signed-rank test and produces data for CD diagrams.

Reference: Demsar, J. (2006). Statistical comparisons of classifiers over
multiple data sets. Journal of Machine Learning Research, 7, 1-30.
"""

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
from itertools import combinations


def wilcoxon_pairwise(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run pairwise Wilcoxon signed-rank tests between all models.
    
    Aggregates results by (dataset, model) first — takes the mean across seeds.
    Returns a DataFrame with columns: model_a, model_b, statistic, p_value, significant.
    """
    # Pivot: rows=datasets, cols=models, values=mean metric
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    
    models = pivot.columns.tolist()
    rows = []
    
    for m1, m2 in combinations(models, 2):
        # Only compare on datasets where both models have results
        paired = pivot[[m1, m2]].dropna()
        if len(paired) < 5:  # need at least 5 datasets for meaningful test
            continue
        
        stat, p = stats.wilcoxon(paired[m1], paired[m2], alternative="two-sided")
        rows.append({
            "model_a": m1,
            "model_b": m2,
            "n_datasets": len(paired),
            "mean_a": paired[m1].mean(),
            "mean_b": paired[m2].mean(),
            "statistic": stat,
            "p_value": p,
            "significant": p < alpha,
            "better": m1 if paired[m1].mean() > paired[m2].mean() else m2,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid comparisons were generated, need a minimum of 5 datasets")
        return df
    
    return pd.DataFrame(rows).sort_values("p_value")


def average_rank_table(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Compute average rank of each model across datasets.
    Lower rank = better (rank 1 = best on a given dataset).
    This is the input needed to draw a Critical Difference diagram.
    """
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    
    # Rank models per dataset (ascending rank = higher metric if higher_is_better)
    if higher_is_better:
        ranked = pivot.rank(axis=1, ascending=False, method="average")
    else:
        ranked = pivot.rank(axis=1, ascending=True, method="average")
    
    avg_ranks = ranked.mean().sort_values()
    rank_df = pd.DataFrame({
        "model": avg_ranks.index,
        "average_rank": avg_ranks.values,
        "n_datasets": (~pivot.isna()).sum().values,
    })
    return rank_df