"""
Data Loading Module
-------------------
Handles loading the Telco Customer Churn dataset from CSV.
"""

import os
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    pd.DataFrame
        Raw DataFrame loaded from the CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
