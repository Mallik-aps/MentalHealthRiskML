from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


PHQ_ITEMS = [f"phq{i}" for i in range(1, 10)]
SRQ_ITEMS = [f"srq{i}" for i in range(1, 21)]


def load_raw_or_synthetic(config: Dict) -> pd.DataFrame:
    raw_path = Path(config["paths"]["raw_dataset"])
    synthetic_path = Path(config["paths"]["synthetic_dataset"])

    if config["data"].get("use_real_if_available", True) and raw_path.exists():
        df = pd.read_csv(raw_path)
    elif synthetic_path.exists():
        df = pd.read_csv(synthetic_path)
    else:
        raise FileNotFoundError(
            "No dataset found. Run src/synthetic_data_generator.py first "
            "or place a real CSV at data/raw/real_multimodal_dataset.csv."
        )

    return harmonize_schema(df)


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_phq = [c for c in PHQ_ITEMS if c not in df.columns]
    missing_srq = [c for c in SRQ_ITEMS if c not in df.columns]
    if missing_phq or missing_srq:
        raise ValueError(f"Missing clinical item columns. PHQ missing={missing_phq}; SRQ missing={missing_srq}")

    if "phq9_total" not in df.columns:
        df["phq9_total"] = df[PHQ_ITEMS].sum(axis=1)
    if "srq20_total" not in df.columns:
        df["srq20_total"] = df[SRQ_ITEMS].sum(axis=1)

    if "risk_label" not in df.columns:
        df["risk_label"] = ((df["phq9_total"] >= 10) | (df["srq20_total"] >= 8)).astype(int)

    return df


def save_integrated_dataset(df: pd.DataFrame, config: Dict) -> str:
    path = Path(config["paths"]["processed_dataset"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)
