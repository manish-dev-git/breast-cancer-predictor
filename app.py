"""
Breast Cancer Predictor Application
-----------------------------------
This Streamlit application predicts whether a breast mass is Malignant or Benign 
based on diagnostic measurements using machine learning models.

Author: [Manish Agrawal]
Version: 1.0.0
Date: [2026-01-02]
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# 1. Load data and models
@st.cache_resource
def load_resources():
    results_df = pd.read_csv('model/results.csv')
    metadata = joblib.load('model/feature_metadata.joblib')
    scaler = joblib.load('model/scaler.joblib')
    return results_df, metadata, scaler

results_df, metadata, scaler = load_resources()
feature_names = metadata['feature_names']
feature_means = metadata['feature_means']

@st.cache_resource
def load_model(model_name):
    model_filename = model_name.lower().replace(' ', '_') + '.joblib'
    model_path = os.path.join('model', model_filename)
    model = joblib.load(model_path)
    return model

# 2. Sidebar for Model Selection
st.sidebar.title("Model Selection")
model_display_names = results_df['Model'].tolist()
selected_model_name = st.sidebar.selectbox("Choose a classifier:", model_display_names)

model = load_model(selected_model_name)

# 3. Sidebar for Sample Data Download
st.sidebar.divider()
st.sidebar.header("Sample Data")
st.sidebar.write("Download a sample dataset to test the application's predictions.")

test_data_file = 'model/test_data.csv'
if os.path.exists(test_data_file):
    with open(test_data_file, "rb") as file:
        st.sidebar.download_button(
            label="Download test_data.csv",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )

# 4. Main Interface
st.title("Breast Cancer Classification App")
st.markdown("""
This application uses machine learning models to predict whether a breast mass is **Malignant** or **Benign** based on diagnostic measurements.
""")

st.header("Select the prediction option:")

# Display radio buttons horizontally
opt = st.radio(
    "",
    ('Batch (Upload Test CSV file)', 'Single (Input data manually)'),
    horizontal=True
)

# with st.expander("Help"):
#     st.write("""
#     - **Batch Prediction**: Upload a CSV file with the required columns for batch predictions.
#     - **Single Prediction**: Enter diagnostic measurements manually for a single prediction.
#     - **Model Selection**: Choose a classifier from the sidebar to use for predictions.
#     """)

if opt == 'Batch (Upload Test CSV file)':
    # Batch prediction logic
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])
    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            missing_features = [col for col in feature_names if col not in test_data.columns]
            if missing_features:
                st.error(f"The uploaded file is missing the following required columns: {missing_features}")
            else:
                st.write("Test Data Uploaded Successfully:")
                # st.dataframe(test_data)
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")

        if st.button("Run Prediction", type="primary"):
            # Preprocess and predict
            # Ensure only the required features are passed to the model
            test_data_features = test_data[feature_names]  # Select only the columns used during training
            input_scaled = scaler.transform(test_data_features)
            predictions = model.predict(input_scaled)
            predictions_proba = model.predict_proba(input_scaled)

            # Display results
            st.subheader("Batch Prediction Results")
            results = test_data.copy()
            results['Prediction'] = predictions
            results['Malignant Probability'] = predictions_proba[:, 1].round(4)
            st.dataframe(results[['target', 'Prediction', 'Malignant Probability']])

            # Option to download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

            malignant_count = results['Prediction'].sum()
            total_count = len(results)
            st.write(f"Malignant Cases: {malignant_count}/{total_count} ({malignant_count / total_count:.2%})")

            # Calculate and display metrics if 'target' column exists
            if 'target' in test_data.columns:
                y_true = test_data['target']
                y_pred = predictions
                y_proba = predictions_proba[:, 1]  # Probability of the positive class

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_proba)
                f1 = f1_score(y_true, y_pred)

                # Display metrics 
                st.subheader("Evaluation Metrics")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"ROC AUC: {roc_auc:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                st.write(cm)

                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.error("The uploaded test file must contain a 'target' column for calculating performance metrics.")

elif opt == 'Single (Input data manually)':
    # Single prediction logic
    st.subheader("Input Diagnostic Measurements")
    input_data = {}
    cols = st.columns(3)  # Display input fields in 3 columns
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            default_val = float(feature_means[name])
            input_data[name] = st.number_input(f"{name.capitalize()}", value=default_val, format="%.4f", key=f"s_{name}")

    if st.button("Run Single Prediction", type="primary"):
        input_array = np.array([list(input_data.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        st.divider()
        st.subheader("Single Prediction Result")
        st.write(f"Prediction: **{'Malignant' if prediction == 1 else 'Benign'}**")
        st.write(f"Malignant Probability: **{prediction_proba[1]:.4f}**")

#Model Performance Comparison
st.divider()
with st.expander("Model Performance Comparison (Training Phase)"):
    # st.header("Model Performance Comparison (Training Phase)")
    st.write("Baseline metrics from model training and evaluation.")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC Score', 'F1 Score'], color='lightgreen'))

