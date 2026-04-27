"""
Model Tuning Module
--------------------
Hyperparameter tuning via GridSearchCV for Random Forest and 
Gradient Boosting classifiers.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer


def tune_random_forest(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune Random Forest using GridSearchCV.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessing transformer.
    X_train, y_train : Training data.
    
    Returns
    -------
    GridSearchCV
        Fitted grid search object.
    """
    print("\n" + "=" * 60)
    print("TUNING RANDOM FOREST")
    print("=" * 60 + "\n")
    
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [8, 12, 16],
        "model__min_samples_split": [5, 10],
        "model__min_samples_leaf": [2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best ROC-AUC (CV): {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_gradient_boosting(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune Gradient Boosting using GridSearchCV.
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessing transformer.
    X_train, y_train : Training data.
    
    Returns
    -------
    GridSearchCV
        Fitted grid search object.
    """
    print("\n" + "=" * 60)
    print("TUNING GRADIENT BOOSTING")
    print("=" * 60 + "\n")
    
    gb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ])
    
    param_grid_gb = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0]
    }
    
    gb_grid = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=param_grid_gb,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    gb_grid.fit(X_train, y_train)
    
    print(f"\nBest Gradient Boosting Parameters: {gb_grid.best_params_}")
    print(f"Best ROC-AUC (CV): {gb_grid.best_score_:.4f}")
    
    return gb_grid
