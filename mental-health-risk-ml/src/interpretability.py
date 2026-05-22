from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_acquisition import load_raw_or_synthetic
from preprocessing import get_feature_names
from utils import ensure_dirs, load_config, safe_filename, set_seed
from validation import validate_dataset


def save_linear_or_tree_importance(model_name, pipeline, feature_names, out_dir):
    model = pipeline.named_steps["model"]
    values = None
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    if values is None:
        return

    df_imp = pd.DataFrame({"feature": feature_names, "importance": values})
    df_imp = df_imp.sort_values("importance", ascending=False)
    df_imp.to_csv(out_dir / f"feature_importance_{safe_filename(model_name)}.csv", index=False)

    top = df_imp.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Top Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir.parent / "figures" / f"feature_importance_{safe_filename(model_name)}.png", dpi=300)
    plt.close(fig)


def try_shap(model_name, pipeline, X_sample, out_fig_dir, out_metrics_dir):
    try:
        import shap
    except Exception as exc:
        print(f"Skipping SHAP for {model_name}: {exc}")
        return

    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    Xt = pre.transform(X_sample)
    feature_names = list(pre.get_feature_names_out())

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xt)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
        else:
            background = shap.sample(Xt, min(100, Xt.shape[0]), random_state=0)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(Xt[: min(80, Xt.shape[0])])
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            Xt = Xt[: min(80, Xt.shape[0])]

        shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_fig_dir / f"shap_summary_{safe_filename(model_name)}.png", dpi=300, bbox_inches="tight")
        plt.close()

        vals = np.abs(shap_values).mean(axis=0)
        imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(out_metrics_dir / f"shap_importance_{safe_filename(model_name)}.csv", index=False)
    except Exception as exc:
        print(f"Could not compute SHAP for {model_name}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    set_seed(config["project"]["random_state"])

    df = load_raw_or_synthetic(config)
    df, _ = validate_dataset(df)
    X = df.drop(columns=["risk_label"])

    sample_size = min(config["interpretability"]["shap_sample_size"], len(X))
    X_sample = X.sample(sample_size, random_state=config["project"]["random_state"])

    model_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])

    for model_path in model_dir.glob("best_*.joblib"):
        model_name = model_path.stem.replace("best_", "").replace("_", " ").title()
        pipeline = joblib.load(model_path)
        feature_names = get_feature_names(pipeline)
        save_linear_or_tree_importance(model_name, pipeline, feature_names, metrics_dir)
        try_shap(model_name, pipeline, X_sample, fig_dir, metrics_dir)

    print("Interpretability outputs saved.")


if __name__ == "__main__":
    main()
