"""
Model Training Module
----------------------
Defines sklearn pipelines for multiple classifiers and provides
a function to compare them all.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from src.evaluation import evaluate_model


def get_pipelines(preprocessor: ColumnTransformer) -> dict:
    """
    Build a dictionary of named sklearn Pipelines for model comparison.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessing transformer.
    
    Returns
    -------
    dict
        {model_name: Pipeline} mapping.
    """
    pipelines = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        
        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ]),
        
        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42))
        ])
    }
    
    return pipelines


def compare_models(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Train and evaluate all pipelines, printing performance metrics.
    
    Parameters
    ----------
    pipelines : dict
        {model_name: Pipeline} mapping.
    X_train, y_train : Training data.
    X_test, y_test : Test data.
    
    Returns
    -------
    dict
        {model_name: {train: metrics_dict, test: metrics_dict}}
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60 + "\n")
    
    all_results = {}
    
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        results = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        all_results[name] = results
    
    return all_results
