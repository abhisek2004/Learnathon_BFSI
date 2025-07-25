import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# --- Page Config ---
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load model and expected columns ---
knn_model = load("knn_fraud_model.pkl")
expected_columns = pd.read_csv("X_train_columns.csv")["columns"].tolist()

# --- Sidebar ---
st.sidebar.header("📊 About")
st.sidebar.markdown(
    """
    **Insurance Fraud Detection App**  
    Upload a CSV file to predict potential fraudulent claims using the trained **KNN model**.
    """
)
st.sidebar.info("Supported file format: CSV")

# --- Main Title ---
st.title("🚨 Insurance Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    uploaded_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(uploaded_data.head(10), use_container_width=True)

    # Separate target column if available
    y_true = None
    if "Fraud_Ind" in uploaded_data.columns:
        y_true = uploaded_data["Fraud_Ind"]
        uploaded_data = uploaded_data.drop(columns=["Fraud_Ind"])

    # Drop unnecessary columns
    drop_cols = ['Claim_ID', 'Policy_Num', 'Vehicle_Registration',
                 'Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date',
                 'Accident_Date', 'Claims_Date', 'DL_Expiry_Date']
    uploaded_data.drop(columns=[col for col in drop_cols if col in uploaded_data.columns],
                       inplace=True, errors='ignore')

    # Fill missing values
    for col in uploaded_data.select_dtypes(include=['float64', 'int64']).columns:
        uploaded_data[col] = uploaded_data[col].fillna(uploaded_data[col].median())
    for col in uploaded_data.select_dtypes(include=['object']).columns:
        uploaded_data[col] = uploaded_data[col].fillna(uploaded_data[col].mode()[0])

    # One-hot encode and align columns
    data_input = pd.get_dummies(uploaded_data)
    for col in expected_columns:
        if col not in data_input.columns:
            data_input[col] = 0
    data_input = data_input[expected_columns]

    # Predict
    y_pred = knn_model.predict(data_input)
    y_prob = knn_model.predict_proba(data_input)[:, 1]

    # Add predictions
    results = uploaded_data.copy()
    results["Fraud_Prediction"] = y_pred
    results["Fraud_Probability"] = y_prob

    st.subheader("Prediction Results")
    st.dataframe(results.head(15), use_container_width=True)

    # --- Metrics and Visualization ---
    if y_true is not None:
        if y_true.dtype == object:
            y_true = y_true.map({'Y': 1, 'N': 0})

        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("ROC-AUC", f"{roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Fraud distribution
        st.subheader("Fraud Prediction Distribution")
        fraud_counts = pd.Series(y_pred).map({0: 'Not Fraud', 1: 'Fraud'}).value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fraud_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], ax=ax2)
        ax2.set_ylabel("Number of Claims")
        st.pyplot(fig2)

    else:
        st.info("No ground truth labels available. Only predictions are displayed.")