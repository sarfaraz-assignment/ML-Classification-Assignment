"""
Streamlit Web Application for ML Classification Model Comparison
Dataset: Breast Cancer Wisconsin (Diagnostic)

This app allows users to:
1. Upload test data (CSV)
2. Select a classification model
3. View evaluation metrics
4. View confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ─────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

FEATURE_COLUMNS = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

TARGET_COLUMN = "target"
TARGET_NAMES = ["Malignant (0)", "Benign (1)"]


# ─────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    """Load a saved model from disk."""
    model_path = os.path.join("model", MODEL_FILES[model_name])
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    """Load the saved StandardScaler."""
    return joblib.load("model/scaler.joblib")


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all 6 evaluation metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# App Header
# ─────────────────────────────────────────────────────────────────────
st.title("🧠 ML Classification Model Dashboard")
st.markdown(
    """
    **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
    **Task:** Binary Classification (Malignant vs Benign)  
    **Features:** 30 numeric features from digitized FNA images  

    Upload your test CSV file, select a model, and explore the evaluation results below.
    """
)
st.divider()

# ─────────────────────────────────────────────────────────────────────
# Sidebar — File Upload & Model Selection
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # CSV Upload
    st.subheader("📂 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with features and target column",
        type=["csv"],
        help="The CSV must contain all 30 feature columns and the 'target' column.",
    )

    st.divider()

    # Model Selection
    st.subheader("🤖 Select Model")
    selected_model = st.selectbox(
        "Choose a classification model",
        list(MODEL_FILES.keys()),
        index=0,
    )

    st.divider()

    # Compare all models toggle
    compare_all = st.checkbox("📊 Compare All Models", value=True)

    st.divider()
    st.caption("Built for BITS WILP M.Tech (AIML) — ML Assignment 2")

# ─────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────

# Load or use uploaded data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded file loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
else:
    # Load default test data
    if os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
        st.info(f"ℹ️ Using default test data: {df.shape[0]} rows × {df.shape[1]} columns. Upload your own CSV from the sidebar.")
    else:
        st.warning("⚠️ No test data found. Please upload a CSV file.")
        st.stop()

# Validate columns
missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
if missing_features:
    st.error(f"❌ Missing feature columns: {missing_features}")
    st.stop()
if TARGET_COLUMN not in df.columns:
    st.error(f"❌ Missing target column: '{TARGET_COLUMN}'")
    st.stop()

# Extract features and target
X_data = df[FEATURE_COLUMNS]
y_data = df[TARGET_COLUMN]

# Show data preview
with st.expander("🔍 Preview Test Data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Number of Features", len(FEATURE_COLUMNS))

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Single Model Evaluation
# ─────────────────────────────────────────────────────────────────────
st.header(f"📋 Results: {selected_model}")

try:
    scaler = load_scaler()
    model = load_model(selected_model)
    X_scaled = scaler.transform(X_data)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    metrics = compute_metrics(y_data, y_pred, y_proba)

    # Metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")

    # Confusion Matrix & ROC Curve side by side
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_data, y_pred, selected_model)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with chart_col2:
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_data, y_proba, selected_model)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # Classification Report
    with st.expander("📄 Full Classification Report", expanded=False):
        report = classification_report(y_data, y_pred, target_names=["Malignant", "Benign"])
        st.code(report, language="text")

except Exception as e:
    st.error(f"❌ Error loading model or computing metrics: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────
# All Models Comparison
# ─────────────────────────────────────────────────────────────────────
if compare_all:
    st.header("📊 All Models — Comparison Table")

    all_results = {}
    scaler = load_scaler()
    X_scaled = scaler.transform(X_data)

    for model_name in MODEL_FILES:
        try:
            mdl = load_model(model_name)
            y_p = mdl.predict(X_scaled)
            y_prob = mdl.predict_proba(X_scaled)[:, 1]
            all_results[model_name] = compute_metrics(y_data, y_p, y_prob)
        except Exception as e:
            st.warning(f"⚠️ Could not evaluate {model_name}: {e}")

    if all_results:
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.index.name = "Model"

        # Style the comparison table — highlight max values
        styled_df = comparison_df.style.highlight_max(axis=0, color="#90EE90").format("{:.4f}")
        st.dataframe(styled_df, use_container_width=True)

        # Bar chart comparison
        st.subheader("📈 Visual Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        comparison_df.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Winner
        best_model_name = comparison_df["F1"].idxmax()
        best_f1 = comparison_df.loc[best_model_name, "F1"]
        st.success(f"🏆 **Overall Winner (by F1 Score):** {best_model_name} with F1 = {best_f1:.4f}")

    # All confusion matrices
    st.subheader("🔢 Confusion Matrices — All Models")
    cm_cols = st.columns(3)
    for idx, model_name in enumerate(MODEL_FILES):
        try:
            mdl = load_model(model_name)
            y_p = mdl.predict(X_scaled)
            with cm_cols[idx % 3]:
                fig_cm = plot_confusion_matrix(y_data, y_p, model_name)
                st.pyplot(fig_cm)
                plt.close(fig_cm)
        except Exception:
            pass

st.divider()
st.caption("© 2026 | BITS WILP M.Tech (AIML) — Machine Learning Assignment 2")
