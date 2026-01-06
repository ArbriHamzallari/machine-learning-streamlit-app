"""Machine learning utilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


def time_split(meta: pd.DataFrame, test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-based train/test split.

    Args:
        meta: DataFrame with date_day column
        test_ratio: Fraction of time for test (default 0.2)

    Returns:
        train_indices, test_indices (numpy arrays of integer positions)
    """
    # Sort unique dates
    unique_dates = sorted(meta["date_day"].unique())
    n_dates = len(unique_dates)
    split_idx = int(n_dates * (1 - test_ratio))

    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_mask = meta["date_day"].isin(train_dates)
    test_mask = meta["date_day"].isin(test_dates)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices


def train_and_eval(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Train and evaluate Logistic Regression and Random Forest.

    Treats community_area as categorical (OneHotEncoded).
    Uses time-based split.
    Selects best model by macro F1.
    Saves metrics to outputs/ml_metrics.json.

    Args:
        X: Feature matrix (must include community_area column)
        y: Target vector (risk_class: 0, 1, 2)
        meta: Metadata with date_day

    Returns:
        best_model, test_predictions, test_probabilities
    """
    # Time-based split
    train_idx, test_idx = time_split(meta, test_ratio=0.2)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # Define categorical and numeric columns
    categorical_cols = ["community_area"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # ColumnTransformer: OneHotEncode community_area, passthrough numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # Fit preprocessor on training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Train Logistic Regression
    lr_model = LogisticRegression(
        max_iter=2000, random_state=42
    )
    lr_model.fit(X_train_transformed, y_train)
    lr_pred = lr_model.predict(X_test_transformed)
    lr_proba = lr_model.predict_proba(X_test_transformed)

    lr_metrics = {
        "model": "LogisticRegression",
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "precision_macro": float(precision_score(y_test, lr_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, lr_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, lr_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
    }

    # Train Random Forest with class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights_dict = dict(zip(classes, class_weights))
    
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        class_weight=weights_dict,
        random_state=42
    )
    rf_model.fit(X_train_transformed, y_train)
    rf_pred = rf_model.predict(X_test_transformed)
    rf_proba = rf_model.predict_proba(X_test_transformed)

    rf_metrics = {
        "model": "RandomForestClassifier",
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "precision_macro": float(precision_score(y_test, rf_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, rf_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, rf_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, rf_pred).tolist(),
    }

    # Select best model by macro F1
    if lr_metrics["f1_macro"] >= rf_metrics["f1_macro"]:
        best_model_name = "LogisticRegression"
        best_model = lr_model
        best_pred = lr_pred
        best_proba = lr_proba
    else:
        best_model_name = "RandomForestClassifier"
        best_model = rf_model
        best_pred = rf_pred
        best_proba = rf_proba

    # Compute binary high-risk metrics
    # Convert to binary: high_risk = (class == 2)
    y_test_binary = (y_test == 2).astype(int)
    best_pred_binary = (best_pred == 2).astype(int)

    binary_high_risk_metrics = {
        "binary_accuracy_high_risk": float(accuracy_score(y_test_binary, best_pred_binary)),
        "precision_high_risk": float(precision_score(y_test_binary, best_pred_binary, zero_division=0)),
        "recall_high_risk": float(recall_score(y_test_binary, best_pred_binary, zero_division=0)),
        "f1_high_risk": float(f1_score(y_test_binary, best_pred_binary, zero_division=0)),
    }

    # Save metrics
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    metrics_output = {
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "best_model": best_model_name,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "binary_high_risk_metrics": binary_high_risk_metrics,
    }

    with open(output_dir / "ml_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)

    # Store preprocessor in best_model for later use
    # (wrap model + preprocessor as a simple object)
    class PipelineWrapper:
        def __init__(self, preprocessor, model):
            self.preprocessor = preprocessor
            self.model = model

        def predict(self, X):
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict(X_transformed)

        def predict_proba(self, X):
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict_proba(X_transformed)

    wrapped_model = PipelineWrapper(preprocessor, best_model)

    return wrapped_model, best_pred, best_proba


if __name__ == "__main__":
    # Quick test
    from src import data_io, features

    df = data_io.filter_year(
        data_io.clean_df(data_io.load_csv("data/chicago_crimes_2025.csv")), 2025
    )
    X, y, meta, _ = features.build_ml_dataset(df)

    model, pred, proba = train_and_eval(X, y, meta)

    print("ML pipeline complete")
    print(f"  Best model saved to outputs/ml_metrics.json")
    print(f"  Test predictions shape: {pred.shape}")
    print(f"  Test probabilities shape: {proba.shape}")
