"""
train.py
Model training, evaluation, comparison, and serialization pipeline.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from data_processing import load_raw_data, clean_data, get_preprocessor, TARGET_COL


def train_and_evaluate(data_path="data/telco_churn.csv", output_dir="models"):
    """
    Train multiple classification models on Telco customer churn,
    select the best model by ROC-AUC, export pipeline and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Loading dataset from {data_path}...")
    df_raw = load_raw_data(data_path)
    df = clean_data(df_raw)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"[*] Dataset shape: {df.shape}. Churn rate: {y.mean():.2%}")

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Candidate Classifiers
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42, min_samples_split=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    }

    comparison_results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None
    best_metrics = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidate_models.items():
        print(f"\n[+] Evaluating {name}...")
        preprocessor = get_preprocessor()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 5-fold cross-validation ROC-AUC
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Train on train set
        pipeline.fit(X_train, y_train)

        # Test set evaluation
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        print(f"    - CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"    - Test Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        comparison_results[name] = {
            "cv_roc_auc_mean": round(cv_mean, 4),
            "cv_roc_auc_std": round(cv_std, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4)
        }

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline
            cm = confusion_matrix(y_test, y_pred).tolist()
            best_metrics = {
                "model_name": name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": cm,
                "total_records": len(df),
                "test_records": len(y_test),
                "churn_rate": round(float(y.mean()), 4)
            }

    print(f"\n[*] Winner Model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

    # Extract feature names and feature importance/coefficients
    feature_importances = []
    try:
        classifier_step = best_pipeline.named_steps['classifier']
        prep_step = best_pipeline.named_steps['preprocessor']
        feature_names = prep_step.get_feature_names_out()

        if hasattr(classifier_step, 'feature_importances_'):
            raw_importances = classifier_step.feature_importances_
            for feat, imp in zip(feature_names, raw_importances):
                clean_name = feat.replace('cat__', '').replace('num__', '')
                feature_importances.append({"feature": clean_name, "importance": round(float(imp), 4)})
        elif hasattr(classifier_step, 'coef_'):
            raw_coefs = np.abs(classifier_step.coef_[0])
            for feat, coef in zip(feature_names, raw_coefs):
                clean_name = feat.replace('cat__', '').replace('num__', '')
                feature_importances.append({"feature": clean_name, "importance": round(float(coef), 4)})

        feature_importances.sort(key=lambda x: x['importance'], reverse=True)
        feature_importances = feature_importances[:12]  # Top 12 features
    except Exception as e:
        print(f"[!] Note: Could not extract feature importances: {e}")

    # Save Best Pipeline
    pipeline_path = os.path.join(output_dir, "churn_pipeline.pkl")
    joblib.dump(best_pipeline, pipeline_path)
    print(f"[*] Serialized model pipeline saved to: {pipeline_path}")

    # Save Metadata & Metrics
    metadata = {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "model_comparisons": comparison_results,
        "top_features": feature_importances,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[*] Model metadata saved to: {metadata_path}")

    return best_pipeline, metadata


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_file = os.path.join(project_root, "data", "telco_churn.csv")
    models_folder = os.path.join(project_root, "models")
    train_and_evaluate(data_file, models_folder)
