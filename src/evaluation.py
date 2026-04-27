"""
Model Evaluation Module
------------------------
Provides functions to evaluate and print model performance metrics.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(
    name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted pipeline on both training and test sets.
    
    Parameters
    ----------
    name : str
        Model name for display.
    pipeline : sklearn Pipeline
        Fitted pipeline with predict and predict_proba methods.
    X_train, y_train : Training data.
    X_test, y_test : Test data.
    
    Returns
    -------
    dict
        Dictionary with train and test metrics.
    """
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Probabilities (for ROC-AUC)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "F1 Score": f1_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred),
        "Recall": recall_score(y_train, y_train_pred),
        "ROC-AUC": roc_auc_score(y_train, y_train_proba),
    }
    
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "F1 Score": f1_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_proba),
    }
    
    # Print results
    print(name)
    print("Model performance for Training set")
    for metric, value in train_metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    print("----------------------------------")
    
    print("Model performance for Test set")
    for metric, value in test_metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    print("=" * 45, "\n")
    
    return {"train": train_metrics, "test": test_metrics}
