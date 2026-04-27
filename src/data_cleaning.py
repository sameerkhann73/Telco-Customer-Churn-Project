"""
Data Cleaning Module
--------------------
Handles data type conversions, missing value imputation,
column dropping, and target encoding.
"""

import pandas as pd


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic data exploration info (mirrors notebook cells 3-7).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    """
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- DataFrame Info ---")
    print(df.info())
    
    print("\n--- Null values per column ---")
    print(df.isnull().sum())
    
    print("\n--- Gender value counts ---")
    print(df['gender'].value_counts())
    
    print("\n--- Descriptive statistics ---")
    print(df.describe())


def clean_data(df: pd.DataFrame):
    """
    Clean the dataset:
    1. Convert TotalCharges to numeric (coerce errors to NaN)
    2. Fill NaN in TotalCharges with median
    3. Drop the customerID column
    4. Separate features (X) and target (y)
    5. Map target to binary (Yes=1, No=0)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        (X, y, cleaned_df) — features, binary target, and the full
        cleaned DataFrame (for EDA before dropping Churn).
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    null_count = df['TotalCharges'].isnull().sum()
    print(f"\nTotalCharges null values after conversion: {null_count}")
    
    # Fill nulls with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)
    
    print("Data cleaning completed successfully.")
    print(f"Final shape: {df.shape}")
    
    # Keep a copy of the cleaned df (including Churn as string) for EDA
    cleaned_df = df.copy()
    
    # Separate features and target
    X = df.drop(['Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"\nTarget distribution:\n{y.value_counts()}")
    
    return X, y, cleaned_df
