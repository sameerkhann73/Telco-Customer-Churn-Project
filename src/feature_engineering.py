"""
Feature Engineering Module
---------------------------
Builds the ColumnTransformer preprocessor that scales numeric features
and one-hot encodes categorical features.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def get_feature_columns(X: pd.DataFrame):
    """
    Identify categorical and numerical feature columns.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (without target).
    
    Returns
    -------
    tuple
        (cat_features, num_features) — Index objects of column names.
    """
    cat_features = X.select_dtypes(include="object").columns
    num_features = X.select_dtypes(exclude="object").columns
    
    print(f"\nCategorical features ({len(cat_features)}): {list(cat_features)}")
    print(f"Numerical features ({len(num_features)}): {list(num_features)}")
    
    return cat_features, num_features


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies StandardScaler to numerical 
    features and OneHotEncoder (drop='first') to categorical features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    
    Returns
    -------
    ColumnTransformer
        Configured preprocessor (unfitted).
    """
    cat_features, num_features = get_feature_columns(X)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
        ]
    )
    
    return preprocessor
