# ML Classification Model Comparison — Breast Cancer Wisconsin (Diagnostic)

## 🎯 Problem Statement

The goal of this project is to build and compare five machine learning classification models to predict whether a breast tumor is **malignant** or **benign** based on features extracted from digitized images of fine needle aspirate (FNA) of breast masses. Early and accurate diagnosis of breast cancer can significantly improve treatment outcomes and patient survival rates.

This is a **binary classification** problem where:
- **Class 0** = Malignant (cancerous)
- **Class 1** = Benign (non-cancerous)

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) / scikit-learn |
| **Instances** | 569 |
| **Features** | 30 (all numeric, real-valued) |
| **Target Classes** | 2 — Malignant (212 samples, 37.3%) and Benign (357 samples, 62.7%) |
| **Missing Values** | None |
| **Train/Test Split** | 80% Train (455) / 20% Test (114), stratified |

### Feature Categories
The 30 features are computed from 10 real-valued measurements of each cell nucleus, with **mean**, **standard error**, and **worst (largest)** values reported for each:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension

All features were standardized using `StandardScaler` before training.

---

## 🔗 GitHub Repository Link

**Repository:** [https://github.com/sarfaraz-bitspilani/ML-Classification-Assignment](https://github.com/sarfaraz-bitspilani/ML-Classification-Assignment)

---

## 🤖 Models Used

Five classification models were implemented and evaluated on the same dataset:

1. **Logistic Regression** — Linear classifier with L2 regularization
2. **Decision Tree Classifier** — Tree-based classifier with max_depth=5
3. **K-Nearest Neighbors (KNN)** — Instance-based learner with k=5
4. **Gaussian Naive Bayes** — Probabilistic classifier assuming feature independence
5. **Support Vector Machine (SVM)** — RBF kernel-based classifier

### 📋 Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

### 🔍 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest overall performance across all metrics (tied with SVM). With 98.25% accuracy, 0.9954 AUC, and an MCC of 0.9623, it demonstrates that a well-tuned linear model can be extremely effective on this dataset. The high AUC (0.9954) indicates excellent discriminative ability between malignant and benign tumors. Only 1 false positive and 1 false negative on the entire test set. This suggests the decision boundary between the two classes is largely linear in the scaled feature space. |
| **Decision Tree** | Lowest performer among all five models with 92.11% accuracy and the lowest AUC (0.9163). The tree had 3 false positives and 6 false negatives — the highest misclassification count. Despite setting max_depth=5 to prevent overfitting, the model struggles with the complexity of 30 features. Decision Trees are prone to axis-aligned splits and may miss diagonal decision boundaries. The MCC of 0.8341 is notably lower, indicating more imbalanced error types. |
| **KNN** | Good performance with 95.61% accuracy and a strong AUC of 0.9788. KNN benefits from the standardized feature space and captures local patterns well. However, it misclassifies 3 benign as malignant and 2 malignant as benign, showing that some samples near the decision boundary are ambiguous. KNN's performance is sensitive to the choice of k and the distance metric; k=5 with Euclidean distance works reasonably well here. |
| **Naive Bayes** | Despite its simplicity and strong independence assumption (which is clearly violated — many features like radius, perimeter, and area are highly correlated), GaussianNB achieves a surprisingly high AUC of 0.9868 (second best). However, its accuracy (92.98%) and MCC (0.8492) are lower, with 4 false positives and 4 false negatives. The high AUC suggests good probability calibration even though the hard predictions have more errors. |
| **Support Vector Machine** | Tied with Logistic Regression for the top spot with identical metrics: 98.25% accuracy, 0.9861 F1, and 0.9623 MCC. The RBF kernel maps features into a higher-dimensional space, effectively finding a nonlinear boundary. The marginally lower AUC (0.9950 vs. 0.9954) is negligible. SVM excels in high-dimensional spaces, which explains its strong performance on this 30-feature dataset. |
| **🏆 Overall Winner** | **Logistic Regression** (tied with SVM). Both achieve identical accuracy (98.25%), F1 (0.9861), and MCC (0.9623). Logistic Regression is preferred as the winner because: (1) it has a marginally higher AUC (0.9954 vs. 0.9950), (2) it is more interpretable — feature coefficients directly indicate feature importance, (3) it trains and infers significantly faster, and (4) it is simpler to deploy and maintain. The strong linear separability of the Breast Cancer dataset in scaled feature space means the added complexity of SVM's RBF kernel provides no additional benefit. |

---

## 🚀 Streamlit App

The interactive Streamlit web application includes:
- ✅ **CSV Upload** — Upload test data with features and target column
- ✅ **Model Selection Dropdown** — Choose from 5 classification models
- ✅ **Evaluation Metrics Display** — Accuracy, AUC, Precision, Recall, F1, MCC
- ✅ **Confusion Matrix** — Heatmap visualization for each model
- ✅ **ROC Curve** — Receiver Operating Characteristic curve
- ✅ **Classification Report** — Detailed per-class precision, recall, and F1
- ✅ **All Models Comparison** — Side-by-side comparison table and bar chart

**Live App:** [Streamlit App Link — Update after deployment]

---

## 📁 Project Structure

```
ML-Classification-Assignment/
│── app.py                  # Streamlit web application
│── train_models.py         # Script to train and evaluate all 5 models
│── requirements.txt        # Python dependencies
│── README.md               # This file
│── test_data.csv           # Test dataset (114 samples)
│── model/
│   ├── scaler.joblib               # StandardScaler (fitted on training data)
│   ├── logistic_regression.joblib  # Trained Logistic Regression model
│   ├── decision_tree.joblib        # Trained Decision Tree model
│   ├── knn.joblib                  # Trained KNN model
│   ├── naive_bayes.joblib          # Trained Naive Bayes model
│   ├── svm.joblib                  # Trained SVM model
│   └── metrics.json                # Pre-computed evaluation metrics
```

---

## 🛠️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/sarfaraz-bitspilani/ML-Classification-Assignment.git
cd ML-Classification-Assignment

# Install dependencies
pip install -r requirements.txt

# (Optional) Re-train the models
python train_models.py

# Run the Streamlit app
streamlit run app.py
```

---

## 📝 Technologies Used

- **Python 3.12**
- **scikit-learn** — ML models and evaluation metrics
- **Streamlit** — Interactive web application
- **pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Matplotlib & Seaborn** — Visualization
- **joblib** — Model serialization

---

*Built for BITS WILP — M.Tech (AIML) — Machine Learning Assignment 2 (2026)*
