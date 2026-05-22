from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from data_acquisition import load_raw_or_synthetic
from evaluate import compute_metrics
from utils import ensure_dirs, load_config, safe_filename
from validation import validate_dataset


def subgroup_report(model_name, estimator, df, attribute):
    rows = []
    X = df.drop(columns=["risk_label"])
    y = df["risk_label"].astype(int)
    for value, idx in df.groupby(attribute).groups.items():
        if len(idx) < 20:
            continue
        Xg = X.loc[idx]
        yg = y.loc[idx]
        pred = estimator.predict(Xg)
        prob = estimator.predict_proba(Xg)[:, 1] if hasattr(estimator, "predict_proba") else None
        m = compute_metrics(yg, pred, prob)
        m.update({"model": model_name, "attribute": attribute, "group": value, "n": len(idx)})
        rows.append(m)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    df = load_raw_or_synthetic(config)
    df, _ = validate_dataset(df)

    rows = []
    model_dir = Path(config["paths"]["models_dir"])
    for model_path in model_dir.glob("best_*.joblib"):
        model_name = model_path.stem.replace("best_", "").replace("_", " ").title()
        est = joblib.load(model_path)
        for attr in ["gender", "education", "occupation", "substance_type"]:
            if attr in df.columns:
                rows.extend(subgroup_report(model_name, est, df, attr))

    out = pd.DataFrame(rows)
    csv_path = Path(config["paths"]["metrics_dir"]) / "fairness_subgroup_metrics.csv"
    out.to_csv(csv_path, index=False)

    if not out.empty:
        for attr in out["attribute"].unique():
            plot = out[out["attribute"] == attr]
            fig, ax = plt.subplots(figsize=(10, 5))
            for model in plot["model"].unique():
                subset = plot[plot["model"] == model]
                ax.plot(subset["group"], subset["sensitivity_recall"], marker="o", label=model)
            ax.set_title(f"Subgroup Sensitivity by {attr}")
            ax.set_ylabel("Sensitivity / Recall")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)
            ax.legend()
            fig.tight_layout()
            fig.savefig(Path(config["paths"]["figures_dir"]) / f"fairness_sensitivity_{safe_filename(attr)}.png", dpi=300)
            plt.close(fig)

    print(f"Saved fairness report: {csv_path}")


if __name__ == "__main__":
    main()
