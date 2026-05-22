from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PHQ_ITEMS = [f"phq{i}" for i in range(1, 10)]
SRQ_ITEMS = [f"srq{i}" for i in range(1, 21)]


def cronbach_alpha(items_df: pd.DataFrame) -> float:
    clean = items_df.dropna()
    if clean.shape[1] < 2 or len(clean) < 3:
        return float("nan")
    item_vars = clean.var(axis=0, ddof=1)
    total_var = clean.sum(axis=1).var(ddof=1)
    k = clean.shape[1]
    if total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def validate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    original_n = len(df)

    if "participant_id" in df.columns:
        df = df.drop_duplicates(subset=["participant_id"])
    else:
        df = df.drop_duplicates()

    rules = [
        df["age"].between(18, 65, inclusive="both"),
        df["phq9_total"].between(0, 27, inclusive="both"),
        df["srq20_total"].between(0, 20, inclusive="both"),
        df["duration_use"].fillna(0) <= df["age"],
        df["age_onset"].fillna(0) <= df["age"],
    ]
    valid_mask = np.logical_and.reduce(rules)
    invalid_rows = int((~valid_mask).sum())
    df = df.loc[valid_mask].reset_index(drop=True)

    report = {
        "original_rows": int(original_n),
        "rows_after_duplicate_removal": int(original_n - (original_n - len(df))),
        "invalid_rows_removed": invalid_rows,
        "final_rows": int(len(df)),
        "missing_values_by_column": df.isna().sum().astype(int).to_dict(),
        "class_distribution": df["risk_label"].value_counts(dropna=False).astype(int).to_dict(),
        "phq9_cronbach_alpha": cronbach_alpha(df[PHQ_ITEMS]),
        "srq20_cronbach_alpha": cronbach_alpha(df[SRQ_ITEMS]),
    }
    return df, report
