"""
Feature Importance Module
--------------------------
Extracts and displays feature importances from a fitted pipeline
(works with tree-based models like GradientBoosting, RandomForest).
"""

import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_importance(best_pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Extract feature importances from a fitted pipeline.
    
    Parameters
    ----------
    best_pipeline : Pipeline
        Fitted sklearn pipeline with 'preprocessor' and 'model' steps.
        The model must have a `feature_importances_` attribute.
    top_n : int
        Number of top features to display.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'Importance'], sorted descending.
    """
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    
    # Get numerical feature names
    num_features = preprocessor.transformers_[0][2]
    
    # Get categorical feature names (after one-hot encoding)
    cat_features = preprocessor.transformers_[1][2]
    ohe = preprocessor.transformers_[1][1]
    ohe_feature_names = ohe.get_feature_names_out(cat_features)
    
    # Combine all feature names
    feature_names = list(num_features) + list(ohe_feature_names)
    
    # Get importances
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    print(f"\nTop {top_n} features:")
    print(feature_importance_df.head(top_n).to_string(index=False))
    
    return feature_importance_df
