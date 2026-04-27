"""
Exploratory Data Analysis Module
---------------------------------
Generates and saves EDA visualizations for churn analysis.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _ensure_plots_dir(plots_dir: str) -> None:
    """Create the plots output directory if it doesn't exist."""
    os.makedirs(plots_dir, exist_ok=True)


def plot_churn_distribution(df: pd.DataFrame, plots_dir: str = "plots") -> None:
    """
    Plot the overall churn distribution (countplot).
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'Churn' column (string values).
    plots_dir : str
        Directory to save the plot.
    """
    _ensure_plots_dir(plots_dir)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "churn_distribution.png"), dpi=150)
    plt.show()


def plot_churn_by_contract(df: pd.DataFrame, plots_dir: str = "plots") -> None:
    """
    Plot churn distribution by contract type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'Contract' and 'Churn' columns.
    plots_dir : str
        Directory to save the plot.
    """
    _ensure_plots_dir(plots_dir)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn', data=df)
    plt.title('Churn by Contract Type')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "churn_by_contract.png"), dpi=150)
    plt.show()


def plot_churn_by_internet_service(df: pd.DataFrame, plots_dir: str = "plots") -> None:
    """
    Plot churn distribution by internet service type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'InternetService' and 'Churn' columns.
    plots_dir : str
        Directory to save the plot.
    """
    _ensure_plots_dir(plots_dir)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='InternetService', hue='Churn', data=df)
    plt.title('Churn by Internet Service')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "churn_by_internet_service.png"), dpi=150)
    plt.show()


def plot_monthly_charges_vs_churn(df: pd.DataFrame, plots_dir: str = "plots") -> None:
    """
    Boxplot of monthly charges by churn status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'MonthlyCharges' and 'Churn' columns.
    plots_dir : str
        Directory to save the plot.
    """
    _ensure_plots_dir(plots_dir)
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title('Monthly Charges vs Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "monthly_charges_vs_churn.png"), dpi=150)
    plt.show()


def run_all_eda(df: pd.DataFrame, plots_dir: str = "plots") -> None:
    """
    Run all EDA visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'Churn' column (string values).
    plots_dir : str
        Directory to save plots.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    plot_churn_distribution(df, plots_dir)
    plot_churn_by_contract(df, plots_dir)
    plot_churn_by_internet_service(df, plots_dir)
    plot_monthly_charges_vs_churn(df, plots_dir)
    
    print(f"\nAll EDA plots saved to '{plots_dir}/' directory.")
