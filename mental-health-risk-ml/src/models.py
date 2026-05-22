from __future__ import annotations

from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def get_models_and_grids(random_state: int) -> Dict[str, Tuple[object, Dict]]:
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=random_state),
            {
                "model__penalty": ["l1", "l2"],
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "SVM RBF": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
            {
                "model__C": [0.5, 1.0, 5.0],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [10, 15, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [5, 10],
                "model__learning_rate": [0.03, 0.05],
            },
        )

    return models
