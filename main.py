"""
Telco Customer Churn Prediction - Main Pipeline
================================================
Orchestrates the full machine learning pipeline:
1. Load data
2. Explore data
3. Clean data
4. Run EDA visualizations
5. Split train/test
6. Build preprocessor
7. Compare baseline models
8. Tune Random Forest & Gradient Boosting
9. Evaluate best models
10. Extract feature importances
"""

import warnings
import os

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.data_cleaning import explore_data, clean_data
from src.eda import run_all_eda
from src.feature_engineering import get_preprocessor
from src.model_training import get_pipelines, compare_models
from src.model_tuning import tune_random_forest, tune_gradient_boosting
from src.evaluation import evaluate_model
from src.feature_importance import get_feature_importance

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    """Run the complete Telco Customer Churn prediction pipeline."""
    
    # ---- 1. Data Loading ----
    data_path = os.path.join("Data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_data(data_path)
    
    # ---- 2. Data Exploration ----
    explore_data(df)
    
    # ---- 3. Data Cleaning ----
    X, y, cleaned_df = clean_data(df)
    
    # ---- 4. EDA Visualizations ----
    run_all_eda(cleaned_df, plots_dir="plots")
    
    # ---- 5. Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # ---- 6. Build Preprocessor ----
    preprocessor = get_preprocessor(X)
    
    # ---- 7. Baseline Model Comparison ----
    pipelines = get_pipelines(preprocessor)
    comparison_results = compare_models(pipelines, X_train, y_train, X_test, y_test)
    
    # ---- 8. Hyperparameter Tuning ----
    # Tune Random Forest
    rf_grid = tune_random_forest(preprocessor, X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    # Tune Gradient Boosting
    gb_grid = tune_gradient_boosting(preprocessor, X_train, y_train)
    best_gb = gb_grid.best_estimator_
    
    # ---- 9. Evaluate Tuned Models ----
    print("\n" + "=" * 60)
    print("TUNED MODEL EVALUATION")
    print("=" * 60 + "\n")
    
    evaluate_model("Tuned Random Forest", best_rf, X_train, y_train, X_test, y_test)
    evaluate_model("Tuned Gradient Boosting", best_gb, X_train, y_train, X_test, y_test)
    
    # ---- 10. Feature Importance (from best GB model) ----
    feature_importance_df = get_feature_importance(best_gb, top_n=15)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
