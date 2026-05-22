from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data_acquisition import load_raw_or_synthetic, save_integrated_dataset
from evaluate import compute_metrics
from models import get_models_and_grids
from preprocessing import build_leakage_safe_pipeline
from utils import ensure_dirs, load_config, safe_filename, save_json, set_seed
from validation import validate_dataset


def _predict_proba_or_score(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    set_seed(config["project"]["random_state"])

    df = load_raw_or_synthetic(config)
    df, validation_report = validate_dataset(df)
    save_integrated_dataset(df, config)
    save_json(validation_report, str(Path(config["paths"]["metrics_dir"]) / "data_validation_report.json"))

    X = df.drop(columns=["risk_label"])
    y = df["risk_label"].astype(int)

    outer_cv = StratifiedKFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=True,
        random_state=config["project"]["random_state"],
    )
    inner_cv = StratifiedKFold(
        n_splits=config["validation"]["inner_cv"],
        shuffle=True,
        random_state=config["project"]["random_state"],
    )

    all_rows = []
    best_params = {}
    roc_data = {}

    models = get_models_and_grids(config["project"]["random_state"])

    for model_name, (model, grid) in models.items():
        print(f"\nTraining {model_name}")
        fold = 0
        for train_idx, test_idx in outer_cv.split(X, y):
            fold += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = build_leakage_safe_pipeline(model, df, config)
            search = GridSearchCV(
                pipe,
                grid,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            y_prob = _predict_proba_or_score(search.best_estimator_, X_test)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics.update({"model": model_name, "fold": fold, "best_score_inner_auc": search.best_score_})
            all_rows.append(metrics)
            best_params.setdefault(model_name, []).append(search.best_params_)

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
            ax.set_title(f"{model_name} - Fold {fold}")
            fig.tight_layout()
            fig.savefig(Path(config["paths"]["figures_dir"]) / f"confusion_matrix_{safe_filename(model_name)}_fold{fold}.png", dpi=300)
            plt.close(fig)

        # Fit final model on all data with inner CV for export.
        final_pipe = build_leakage_safe_pipeline(model, df, config)
        final_search = GridSearchCV(final_pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, refit=True)
        final_search.fit(X, y)
        joblib.dump(final_search.best_estimator_, Path(config["paths"]["models_dir"]) / f"best_{safe_filename(model_name)}.joblib")

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = Path(config["paths"]["metrics_dir"]) / "cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = metrics_df.groupby("model").agg(["mean", "std"])
    summary.to_csv(Path(config["paths"]["metrics_dir"]) / "cv_metrics_summary.csv")
    save_json(best_params, str(Path(config["paths"]["metrics_dir"]) / "best_params.json"))

    # Model comparison plot.
    plot_df = metrics_df.groupby("model")[["roc_auc", "accuracy", "sensitivity_recall", "specificity", "f1"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validated Model Performance")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(config["paths"]["figures_dir"]) / "model_comparison_auc.png", dpi=300)
    plt.close(fig)

    # ROC curves from final models on training data for visual sanity check.
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in models:
        est = joblib.load(Path(config["paths"]["models_dir"]) / f"best_{safe_filename(model_name)}.joblib")
        RocCurveDisplay.from_estimator(est, X, y, ax=ax, name=model_name)
    ax.set_title("ROC Curves of Final Fitted Baselines")
    fig.tight_layout()
    fig.savefig(Path(config["paths"]["figures_dir"]) / "roc_curves.png", dpi=300)
    plt.close(fig)

    print(f"\nSaved metrics to {metrics_path}")
    print(summary)


if __name__ == "__main__":
    main()
