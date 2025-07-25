import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load your trained model
knn_model = load("knn_fraud_model.pkl")

# Load expected columns
expected_columns = pd.read_csv("X_train_columns.csv")["columns"].tolist()

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")
st.title("🚨 Insurance Fraud Detection Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file for fraud prediction", type=["csv"])

if uploaded_file:
    # Read uploaded data
    uploaded_data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(uploaded_data.head())

    # Keep copy of true labels if available
    y_true = None
    if "Fraud_Ind" in uploaded_data.columns:
        y_true = uploaded_data["Fraud_Ind"]
        uploaded_data = uploaded_data.drop(columns=["Fraud_Ind"])

    # Drop unnecessary columns
    drop_cols = ['Claim_ID', 'Policy_Num', 'Vehicle_Registration',
                 'Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date',
                 'Accident_Date', 'Claims_Date', 'DL_Expiry_Date']
    for col in drop_cols:
        if col in uploaded_data.columns:
            uploaded_data = uploaded_data.drop(columns=[col])

    # Fill missing values
    for col in uploaded_data.select_dtypes(include=['float64', 'int64']).columns:
        uploaded_data[col] = uploaded_data[col].fillna(uploaded_data[col].median())
    for col in uploaded_data.select_dtypes(include=['object']).columns:
        uploaded_data[col] = uploaded_data[col].fillna(uploaded_data[col].mode()[0])

    # One-hot encode
    data_input = pd.get_dummies(uploaded_data)

    # Align columns with expected
    for col in expected_columns:
        if col not in data_input.columns:
            data_input[col] = 0
    data_input = data_input[expected_columns]

    # Predict
    y_pred = knn_model.predict(data_input)
    y_prob = knn_model.predict_proba(data_input)[:, 1]

    # Add predictions to data
    results = uploaded_data.copy()
    results["Fraud_Prediction"] = y_pred
    results["Fraud_Probability"] = y_prob

    st.subheader("Prediction Results")
    st.dataframe(results.head(10))

    # Show metrics if ground truth is available
    if y_true is not None:
        # Convert 'Y'/'N' to 1/0 if needed
        if y_true.dtype == object:
            y_true = y_true.map({'Y': 1, 'N': 0})

        acc = (y_true == y_pred).mean()
        st.metric("Accuracy", f"{acc:.4f}")

        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        roc_auc = roc_auc_score(y_true, y_prob)
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
