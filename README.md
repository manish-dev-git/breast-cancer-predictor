# Breast Cancer Predictor Application

This application predicts whether a breast mass is **Malignant** or **Benign** based on diagnostic measurements using machine learning models. It uses the Breast Cancer Wisconsin (Diagnostic) dataset and evaluates multiple machine learning models for classification.

---

## Model Performance Summary

| Model                | Accuracy  | AUC Score | Precision | Recall   | F1 Score | MCC      |
|----------------------|-----------|-----------|-----------|----------|----------|----------|
| Logistic Regression  | 0.973684 | 0.997380  | 0.957746  | 0.957746 | 0.957746 | 0.887979 |
| K-Nearest Neighbor   | 0.947368 | 0.981985  | 0.957746  | 0.957746 | 0.957746 | 0.887979 |
| Gaussian Naive Bayes | 0.964912 | 0.997380  | 0.958904  | 0.985915 | 0.972222 | 0.925285 |
| Random Forest        | 0.964912 | 0.995251  | 0.958904  | 0.985915 | 0.972222 | 0.925285 |
| XGBoost              | 0.956140 | 0.990829  | 0.958333  | 0.971831 | 0.965035 | 0.906379 |

---

## Key Observations

### Logistic Regression
- **Top Performer**: Achieved an accuracy of approximately **97.37%** and an area under the curve (AUC) score of **0.997**.
- **Reason**: Its superior performance suggests that the features in this dataset have a strong linear relationship with the target variable.

### Decision Tree
- **Baseline Performance**: Provides a good baseline for comparison with other models.

### K-Nearest Neighbor
- **Performance**: Achieved an accuracy of **94.74%** with an AUC score of **0.982**.
- **Observation**: Sensitive to the choice of `k` and the scaling of features.

### Gaussian Naive Bayes
- **Strength**: Performed well with an accuracy of **96.49%** and an AUC score of **0.997**.
- **Observation**: Assumes feature independence, which works well for this dataset.

### Random Forest
- **Robust Model**: Achieved an accuracy of **96.49%** with an AUC score of **0.995**.
- **Strength**: Handles feature interactions and is less prone to overfitting.

### XGBoost
- **Performance**: Achieved an accuracy of **95.61%** with an AUC score of **0.991**.
- **Strength**: Performs well on structured datasets and handles missing data effectively.

---

## How to Use the Application

1. **Batch Prediction**:
   - Upload a CSV file containing diagnostic measurements.
   - The app will preprocess the data and predict whether each sample is Malignant or Benign.

2. **Single Prediction**:
   - Manually input diagnostic measurements.
   - The app will predict whether the sample is Malignant or Benign.

---

## Requirements

- Python 3.8+
- Libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `joblib`
  - `scikit-learn`

---


