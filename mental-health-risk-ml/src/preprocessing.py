from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numerical columns to IQR-derived bounds learned from training data only."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(arr, 25, axis=0)
        self.q3_ = np.nanpercentile(arr, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.multiplier * iqr
        self.upper_ = self.q3_ + self.multiplier * iqr
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


def split_columns(df: pd.DataFrame, target: str = "risk_label") -> Tuple[List[str], List[str]]:
    excluded = {target, "participant_id"}
    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in df.columns
        if c not in excluded and not pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols, categorical_cols


def build_preprocessor(df: pd.DataFrame, config: dict) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_columns(df)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if config["preprocessing"].get("winsorize_outliers", True):
        numeric_steps.append(("winsor", IQRWinsorizer(config["preprocessing"].get("iqr_multiplier", 1.5))))
    numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_leakage_safe_pipeline(model, df: pd.DataFrame, config: dict) -> ImbPipeline:
    preprocessor = build_preprocessor(df, config)
    steps = [("preprocessor", preprocessor)]
    if config["preprocessing"].get("use_smote", True):
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=config["project"]["random_state"],
                    k_neighbors=config["preprocessing"].get("smote_k_neighbors", 5),
                ),
            )
        )
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def get_feature_names(fitted_pipeline) -> List[str]:
    pre = fitted_pipeline.named_steps["preprocessor"]
    return list(pre.get_feature_names_out())
