from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils import ensure_dirs, load_config, set_seed


def _choice(rng, values, probs, n):
    return rng.choice(values, size=n, p=np.array(probs) / np.sum(probs))


def generate_synthetic_multimodal_dataset(config: Dict) -> pd.DataFrame:
    seed = int(config["project"]["random_state"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    n = int(config["data"]["n_samples"])
    target_high_risk_ratio = float(config["data"]["high_risk_ratio"])

    age = np.clip(rng.normal(34.2, 9.5, n).round(), 18, 65).astype(int)
    gender = _choice(rng, ["Male", "Female", "Other"], [0.62, 0.36, 0.02], n)
    education = _choice(rng, ["Primary", "Secondary", "Higher"], [0.30, 0.45, 0.25], n)
    occupation = _choice(rng, ["Unemployed", "Informal", "Formal", "Student"], [0.29, 0.34, 0.27, 0.10], n)
    marital_status = _choice(rng, ["Single", "Married", "Separated", "Widowed"], [0.42, 0.42, 0.12, 0.04], n)

    substance_type = _choice(rng, ["Alcohol", "Cannabis", "Opioids", "Others", "Polysubstance"], [0.42, 0.21, 0.17, 0.08, 0.12], n)
    frequency_use = np.clip(rng.normal(3.4, 1.2, n), 1, 7).round(1)
    duration_use = np.clip(rng.gamma(shape=2.2, scale=3.6, size=n), 1, 25).round(1)
    duration_use = np.minimum(duration_use, np.maximum(age - 12, 1))
    age_onset = np.clip(age - duration_use + rng.normal(0, 2.0, n), 12, age).round(1)

    risk_score = (
        -2.2
        + 0.04 * (age - 30)
        + 0.35 * (frequency_use - 3)
        + 0.07 * duration_use
        + 0.55 * np.isin(substance_type, ["Opioids", "Polysubstance"]).astype(float)
        + 0.25 * (education == "Primary").astype(float)
        + 0.30 * (occupation == "Unemployed").astype(float)
        + rng.normal(0, 0.7, n)
    )
    prob = 1 / (1 + np.exp(-risk_score))
    threshold = np.quantile(prob, 1 - target_high_risk_ratio)
    risk_label = (prob >= threshold).astype(int)

    phq_items = {}
    srq_items = {}
    for i in range(1, 10):
        lam = np.where(risk_label == 1, rng.uniform(1.25, 2.35), rng.uniform(0.25, 1.1))
        phq_items[f"phq{i}"] = np.clip(rng.poisson(lam=lam), 0, 3)
    for i in range(1, 21):
        p = np.where(risk_label == 1, rng.uniform(0.42, 0.75), rng.uniform(0.08, 0.38))
        srq_items[f"srq{i}"] = rng.binomial(1, p, n)

    df = pd.DataFrame(
        {
            "participant_id": [f"P{i:05d}" for i in range(1, n + 1)],
            "sample_source": np.where(rng.random(n) < float(config["data"]["synthetic_ratio"]), "synthetic", "real_like"),
            "age": age,
            "gender": gender,
            "education": education,
            "occupation": occupation,
            "marital_status": marital_status,
            "substance_type": substance_type,
            "frequency_use": frequency_use,
            "duration_use": duration_use,
            "age_onset": age_onset,
            **phq_items,
            **srq_items,
        }
    )
    df["phq9_total"] = df[[f"phq{i}" for i in range(1, 10)]].sum(axis=1)
    df["srq20_total"] = df[[f"srq{i}" for i in range(1, 21)]].sum(axis=1)

    # Label uses clinical thresholds and risk score; this simulates diagnostic labels with realistic association.
    df["risk_label"] = ((df["phq9_total"] >= 10) | (df["srq20_total"] >= 8) | (risk_label == 1)).astype(int)

    # Introduce light missingness in non-critical variables to test pipeline.
    for col in ["education", "occupation", "frequency_use", "duration_use", "phq3", "srq5"]:
        mask = rng.random(n) < 0.025
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    df = generate_synthetic_multimodal_dataset(config)
    out = Path(config["paths"]["synthetic_dataset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    processed = Path(config["paths"]["processed_dataset"])
    processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed, index=False)
    print(f"Saved dataset: {out}")
    print(f"Saved integrated copy: {processed}")
    print(df["risk_label"].value_counts(normalize=True).rename("class_ratio"))


if __name__ == "__main__":
    main()
