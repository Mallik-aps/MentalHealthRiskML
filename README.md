# Mental Health Risk Prediction using Multimodal Machine Learning

### A Comprehensive Explainable AI Framework for Early Detection of Depression and Common Mental Disorders Using Multimodal Clinical and Behavioral Data

---

# Overview

This project presents a comprehensive multimodal machine learning framework for early detection of depression and common mental disorders (CMDs) among psychoactive substance users. The framework integrates socio-demographic, behavioral, and clinical assessment data using explainable AI, leakage-safe validation, baseline machine learning models, fairness analysis, and reproducible preprocessing pipelines for clinically interpretable mental health risk prediction.

The repository provides a fully reproducible research pipeline including:

- Multimodal data integration  
- Data preprocessing and quality assurance  
- Synthetic data augmentation  
- Leakage-safe stratified cross-validation  
- Baseline machine learning models  
- Explainable AI (SHAP + LIME)  
- Fairness and bias evaluation  
- Publication-quality visualizations  
- Modular GitHub-ready architecture  

---

# Key Features

- Multimodal feature fusion for mental health prediction  
- Leakage-safe preprocessing and validation pipeline  
- Baseline ML models:
  - Logistic Regression
  - Support Vector Machine (RBF)
  - Random Forest
  - XGBoost
- Explainable AI using SHAP and LIME  
- Fairness analysis across demographic subgroups  
- Automated visualization and reporting  
- Reproducible configuration-driven pipeline  

---

# Repository Structure

```text
mental-health-risk-ml/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── .gitignore
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── src/
│   ├── data_acquisition.py
│   ├── synthetic_data_generator.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── models.py
│   ├── train_cv.py
│   ├── evaluate.py
│   ├── interpretability.py
│   ├── fairness_analysis.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_dataset_generation.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baseline_model_training.ipynb
│   ├── 04_model_interpretability.ipynb
│   └── 05_result_visualization.ipynb
│
├── outputs/
│   ├── metrics/
│   ├── figures/
│   ├── models/
│   └── reports/
│
└── tests/
    ├── test_preprocessing.py
    ├── test_metrics.py
    └── test_data_validation.py
```

---

# Dataset Description

The framework supports multimodal datasets containing:

## 1. Socio-Demographic Features

Examples:

- Age  
- Gender  
- Education  
- Occupation  
- Marital status  

---

## 2. Behavioral / Substance Use Features

Examples:

- Substance type  
- Frequency of use  
- Duration of use  
- Age of onset  
- Polysubstance usage  

---

## 3. Clinical Assessment Features

### PHQ-9

Patient Health Questionnaire for depression severity.

Range:

```text
0–27
```

### SRQ-20

Self Reporting Questionnaire for CMD screening.

Range:

```text
0–20
```

---

# Methodology Pipeline

## Step 1 — Data Acquisition

The system integrates multimodal participant records from:

- Clinical records  
- Public mental health datasets  
- Community surveys  
- Synthetic augmentation procedures  

---

## Step 2 — Data Validation

Validation checks include:

- Duplicate removal  
- Feature range verification  
- Consistency rules  
- Missing-value analysis  
- Reliability testing  

---

## Step 3 — Data Preprocessing

The preprocessing pipeline performs:

- Missing value imputation  
- Z-score normalization  
- One-hot encoding  
- Outlier handling  
- SMOTE balancing  

---

## Step 4 — Leakage-Safe Cross Validation

A strict stratified 5-fold cross-validation protocol is used.

Important safeguards:

- Preprocessing is applied ONLY to training folds  
- Validation/test folds use learned parameters only  
- SMOTE is performed only within training folds  

This prevents:

- Data leakage  
- Target leakage  
- Over-optimistic performance estimates  

---

## Step 5 — Model Training

Implemented baseline models:

| Model | Description |
|---|---|
| Logistic Regression | Linear interpretable classifier |
| SVM (RBF) | Nonlinear kernel classifier |
| Random Forest | Ensemble tree-based classifier |
| XGBoost | Gradient boosting framework |

---

## Step 6 — Evaluation

Evaluation metrics include:

- Accuracy  
- Precision  
- Recall  
- Specificity  
- F1-score  
- ROC-AUC  
- PR-AUC  

---

## Step 7 — Explainability

Implemented interpretability techniques:

| Method | Purpose |
|---|---|
| SHAP | Global and local feature contribution |
| LIME | Local explanation of predictions |
| Feature Importance | Tree-based interpretability |

---

# Installation

Clone repository

```bash
git clone https://github.com/your-username/mental-health-risk-ml.git

cd mental-health-risk-ml
```

---

# Create Virtual Environment

Using Conda

```bash
conda create -n mental-health python=3.10

conda activate mental-health
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:

- numpy  
- pandas  
- scikit-learn  
- imbalanced-learn  
- xgboost  
- shap  
- lime  
- matplotlib  
- seaborn  
- joblib  
- pyyaml  
- pytest  

---

# Running the Project

## 1. Generate Synthetic Dataset

```bash
python src/synthetic_data_generator.py
```

Generated dataset:

```text
data/synthetic/
```

---

## 2. Train Models

```bash
python src/train_cv.py
```

Outputs:

- Trained models  
- Cross-validation metrics  
- ROC curves  
- Evaluation reports  

---

## 3. Evaluate Models

```bash
python src/evaluate.py
```

---

## 4. Run Interpretability Analysis

```bash
python src/interpretability.py
```

Generated outputs:

- SHAP summary plots  
- SHAP dependence plots  
- Feature importance rankings  
- LIME explanations  

---

## 5. Run Fairness Analysis

```bash
python src/fairness_analysis.py
```

Outputs:

- Gender-wise evaluation  
- Age-group analysis  
- Education-group fairness metrics  

---

# Output Directory

Generated outputs are stored in:

```text
outputs/
```

## Metrics

```text
outputs/metrics/
```

Contains:

- CSV metric reports  
- Fold-wise evaluation summaries  

---

## Figures

```text
outputs/figures/
```

Contains:

- ROC curves  
- Confusion matrices  
- SHAP plots  
- Feature importance plots  
- Fairness visualizations  

---

## Models

```text
outputs/models/
```

Contains serialized trained models:

```text
.joblib
.pkl
```

---

# Ethical and Data Governance Considerations

This repository supports:

- Anonymized data handling  
- Synthetic augmentation  
- Reproducible preprocessing  
- Fairness analysis  
- Transparent model explanations  

No personally identifiable information should be included in uploaded datasets.

---

# Reproducibility

The repository ensures reproducibility through:

- Fixed random seeds  
- Configuration-driven execution  
- Version-controlled preprocessing  
- Modular architecture  
- Deterministic pipelines  

---

# Future Extensions

Potential future improvements include:

- Deep learning architectures  
- Transformer-based multimodal fusion  
- Temporal mental health prediction  
- Federated learning integration  
- Real-time clinical decision support  
- Explainable deep neural networks  

---

# Citation

```
If you are using this repository or the corresponding research work, please cite the following paper:
@article{MentalHealthRiskML2026,
title={A Multimodal Machine Learning Framework for Early Detection of Depression and Mental Disorders in Psychoactive Substance Users},
year={2026}
}
```

---

# License

This project is licensed under the MIT License. See the LICENSE file for details.

---

