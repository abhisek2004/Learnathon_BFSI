import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Title and Description
st.title("📊 Insurance Fraud Detection - EDA Dashboard")
st.markdown("""
Welcome to the Fraud Detection Exploratory Dashboard!  
Here, you can explore patterns in your dataset interactively.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your cleaned CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    # Select Columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.sidebar.subheader("Visualize Data")
    plot_type = st.sidebar.selectbox("Choose plot type", ["Histogram", "Boxplot", "Correlation Heatmap", "Bar Chart", "Scatter Plot"])

    if plot_type == "Histogram":
        col = st.sidebar.selectbox("Select column for histogram", numeric_cols)
        bins = st.sidebar.slider("Number of bins", 5, 100, 20)
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Boxplot":
        col = st.sidebar.selectbox("Select column for boxplot", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    elif plot_type == "Correlation Heatmap":
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif plot_type == "Bar Chart":
        col = st.sidebar.selectbox("Select categorical column", categorical_cols)
        fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col,
                     labels={'index': col, col: 'Count'}, title=f'Bar chart of {col}')
        st.plotly_chart(fig)

    elif plot_type == "Scatter Plot":
        col_x = st.sidebar.selectbox("X axis", numeric_cols)
        col_y = st.sidebar.selectbox("Y axis", numeric_cols)
        hue = st.sidebar.selectbox("Color by", categorical_cols + ["None"])
        fig = px.scatter(df, x=col_x, y=col_y, color=hue if hue != "None" else None,
                         title=f"{col_x} vs {col_y}")
        st.plotly_chart(fig)

    # Target variable info
    if "fraud_reported" in df.columns:
        st.markdown("### 📌 Target Distribution: Fraud Reported")
        fig = px.pie(df, names='fraud_reported', title='Fraud vs Non-Fraud Cases')
        st.plotly_chart(fig)
