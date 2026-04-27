# Telco Customer Churn Prediction

A modular machine learning project to predict customer churn for a telecommunications company using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset.

## Project Structure

```
Telco_Customer_Churn_project/
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── plots/                                       # Generated EDA visualizations
├── src/
│   ├── __init__.py                              # Package init
│   ├── data_loader.py                           # Data loading
│   ├── data_cleaning.py                         # Data exploration & cleaning
│   ├── eda.py                                   # EDA visualizations
│   ├── feature_engineering.py                   # Preprocessor (scaling + encoding)
│   ├── model_training.py                        # Baseline model comparison
│   ├── model_tuning.py                          # Hyperparameter tuning (GridSearchCV)
│   ├── evaluation.py                            # Metrics evaluation
│   └── feature_importance.py                    # Feature importance extraction
├── main.py                                      # Orchestrator script
├── requirements.txt                             # Dependencies
└── README.md                                    # This file
```

## Pipeline Overview

1. **Data Loading** — Load the CSV dataset
2. **Data Exploration** — Print basic stats, null counts, value distributions
3. **Data Cleaning** — Convert `TotalCharges` to numeric, fill NaNs, drop `customerID`, encode target
4. **EDA** — Generate 4 visualizations (churn distribution, churn by contract, churn by internet service, monthly charges boxplot)
5. **Feature Engineering** — `StandardScaler` for numeric features, `OneHotEncoder` for categoricals via `ColumnTransformer`
6. **Baseline Model Comparison** — Train & evaluate Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
7. **Hyperparameter Tuning** — `GridSearchCV` for Random Forest and Gradient Boosting
8. **Evaluation** — Accuracy, F1, Precision, Recall, ROC-AUC on train and test sets
9. **Feature Importance** — Top 15 features from the best Gradient Boosting model

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python main.py
```

## Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Single tree classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosted trees |

## Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7,043 customers
- **Columns**: 21 features including demographics, services, account info, and churn status
