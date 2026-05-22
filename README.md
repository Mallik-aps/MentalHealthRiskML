# Mental Health Risk ML

A complete, reproducible Python implementation for early detection of depression and common mental disorders (CMD) among psychoactive substance users using multimodal socio-demographic, substance-use, and PHQ-9/SRQ-20 clinical features.

The implementation follows a leakage-safe design:

- imputation, encoding, scaling, outlier handling, and SMOTE are fitted only on training folds
- stratified cross-validation is used for class-balanced evaluation
- hyperparameter tuning is performed inside training folds
- metrics are exported as CSV/JSON
- SHAP/LIME interpretability and fairness subgroup reports are generated

> This repository is for methodological research and baseline benchmarking only. It is not a clinical diagnostic system.

## Repository Structure

```text
config/config.yaml                 Experiment settings
data/raw/                           Optional real CSV files
data/processed/                     Generated processed outputs
src/                               Main source code
notebooks/                         Lightweight starter notebooks
outputs/metrics/                   Results CSV/JSON files
outputs/figures/                   Evaluation and interpretation plots
outputs/models/                    Saved fitted models
tests/                             Basic tests
```

## Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

python src/synthetic_data_generator.py --config config/config.yaml
python src/train_cv.py --config config/config.yaml
python src/interpretability.py --config config/config.yaml
python src/fairness_analysis.py --config config/config.yaml
```

## Optional: Use Your Own Dataset

Place a CSV file at:

```text
data/raw/real_multimodal_dataset.csv
```

The expected columns are:

```text
age, gender, education, occupation, marital_status,
substance_type, frequency_use, duration_use, age_onset,
phq1, phq2, phq3, phq4, phq5, phq6, phq7, phq8, phq9,
srq1, ..., srq20,
risk_label
```

If `phq9_total` and `srq20_total` are not provided, they are computed automatically.

## Main Outputs

After running the pipeline, check:

```text
outputs/metrics/cv_metrics.csv
outputs/metrics/cv_metrics_summary.csv
outputs/metrics/best_params.json
outputs/figures/model_comparison_auc.png
outputs/figures/roc_curves.png
outputs/figures/confusion_matrix_*.png
outputs/figures/shap_summary_*.png
outputs/figures/fairness_*.png
outputs/models/best_*.joblib
```

## Important Methodological Notes

SMOTE is applied through `imblearn.Pipeline`, so synthetic samples are created only within training folds. Validation/test folds are transformed using preprocessing parameters learned from the training portion only.

## License

MIT License.
