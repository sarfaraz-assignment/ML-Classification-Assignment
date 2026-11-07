"""
Train all 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluate them, save the models and test data, and print comparison metrics.

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository (via scikit-learn)
Features: 30 numeric features computed from digitized images of fine needle aspirates (FNA)
Instances: 569
Classes: 2 (Malignant = 0, Benign = 1)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import joblib

# ─────────────────────────────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"Dataset shape: {X.shape}")
print(f"Classes: {dict(enumerate(data.target_names))}")
print(f"Class distribution:\n{y.value_counts()}\n")

# ─────────────────────────────────────────────────────────────────────
# 2. Train/Test Split (80/20, stratified)
# ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}\n")

# ─────────────────────────────────────────────────────────────────────
# 3. Feature Scaling
# ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")
print("Scaler saved to model/scaler.joblib")

# ─────────────────────────────────────────────────────────────────────
# 4. Save Test Data as CSV (with target column)
# ─────────────────────────────────────────────────────────────────────
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Test data saved to test_data.csv ({test_df.shape[0]} rows, {test_df.shape[1]} columns)\n")

# ─────────────────────────────────────────────────────────────────────
# 5. Define Models
# ─────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}

model_filenames = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# ─────────────────────────────────────────────────────────────────────
# 6. Train, Evaluate, and Save Each Model
# ─────────────────────────────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    }

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

    # Save model
    model_path = os.path.join("model", model_filenames[name])
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}\n")

# ─────────────────────────────────────────────────────────────────────
# 7. Save metrics to JSON for the Streamlit app
# ─────────────────────────────────────────────────────────────────────
with open("model/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Metrics saved to model/metrics.json")

# ─────────────────────────────────────────────────────────────────────
# 8. Print Comparison Table
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("COMPARISON TABLE")
print(f"{'='*80}")
comparison_df = pd.DataFrame(results).T
print(comparison_df.to_string())

# Find overall winner
print(f"\n{'='*80}")
best_model = comparison_df["F1"].idxmax()
print(f"Overall Winner (by F1 Score): {best_model} (F1 = {comparison_df.loc[best_model, 'F1']:.4f})")
print(f"{'='*80}")

