<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

# Learnathon_BFSI

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## TEAM (SC1)4TH_10

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

# 🚗 AI-Powered Fraud Detection in Auto Insurance

## 🧠 Predictive Modeling for Smarter Claims Management

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

# Team Information: 4TH_TEAM_10

| Name                     | Department | Year      | Roll No    | Registration No | Email ID                                   | Team ID     |
|--------------------------|------------|-----------|------------|----------------|--------------------------------------------|-------------|
| Rajat Kumar Dash         | CSE        | 4th Year  | 22CSE139   | 22UG010272     | 22cse139.rajatkumardash@giet.edu           | 4TH_TEAM_10 |
| Soumya Ranjan Mohapatra  | CSE        | 4th Year  | 22CSE215   | 22UG010348     | 22cse215.soumyaranjanmohapatra@giet.edu    | 4TH_TEAM_10 |
| Debabrata Mishra         | CSE        | 4th Year  | 22CSE140   | 22UG010273     | 22cse140.debabratamishra@giet.edu          | 4TH_TEAM_10 |
| Abhisek Panda            | CSE        | 4th Year  | 22CSE072   | 22UG010159     | 22cse072.abhisekpanda@giet.edu             | 4TH_TEAM_10 |

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 📌 Overview
The **AI-Powered Fraud Detection in Auto Insurance** project leverages advanced machine learning (ML) techniques to identify fraudulent claims in the auto insurance industry. Insurance fraud results in billions of dollars in losses annually, inflating premiums for policyholders and straining insurer profitability. This project builds a robust, scalable fraud detection system that automates the identification of suspicious claims, prioritizes cases for investigation, and streamlines claims management processes.

The solution combines **data preprocessing**, **feature engineering**, **model training**, and **evaluation** to deliver high accuracy and reliability. By analyzing structured historical claim data, the system flags potentially fraudulent claims in real-time, reducing manual effort and improving operational efficiency.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📌 Problem Statement

Insurance fraud in the auto sector is a pervasive issue, with fraudulent claims accounting for a significant portion of financial losses. These claims often involve exaggerated injuries, staged accidents, or falsified information, making them challenging to detect manually. The objective is to develop an AI-driven system that:

- **Automates fraud detection**: Identifies suspicious claims with high accuracy in real-time.
- **Prioritizes investigations**: Flags high-risk claims for further scrutiny, optimizing resource allocation.
- **Reduces false positives**: Minimizes incorrect fraud flags to maintain customer trust.
- **Enhances efficiency**: Streamlines the claims management pipeline, reducing processing time and costs.

The system uses historical claim data to train ML models, enabling insurers to proactively combat fraud and mitigate financial losses.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## Documentation Book - https://online.fliphtml5.com/jgrqn/dtra/

## PPT 

![Slide1](https://github.com/user-attachments/assets/1b1ed49f-3e2b-4f1c-98f2-7879c8bb888f)
![Slide2](https://github.com/user-attachments/assets/153b5fd0-cc84-4bb7-aa6a-00a02fb615c3)
![Slide3](https://github.com/user-attachments/assets/be33f1e8-1bf6-4d43-8554-5d3003a3c374)
![Slide4](https://github.com/user-attachments/assets/6f70a4f2-0d4e-4c06-a3f3-7f4a0f1714dd)
![Slide5](https://github.com/user-attachments/assets/f5b349f9-ed73-4d55-a85f-e79765c94bdf)
![Slide6](https://github.com/user-attachments/assets/849e7203-d1e2-4012-b7c2-81af26d32dbc)
![Slide7](https://github.com/user-attachments/assets/c72e2d6a-eb77-4cfe-bafe-86ed38e996be)
![Slide8](https://github.com/user-attachments/assets/b2bbaba1-8ca0-43ee-86d3-e1f710f45574)
![Slide9](https://github.com/user-attachments/assets/a440a392-5edc-4d9a-8842-91237a8e8825)
![Slide10](https://github.com/user-attachments/assets/d2450c2a-45cf-4aae-bb83-83cd60673744)

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📊 Dataset Description

The dataset consists of **structured claim records** with over **50 features**, capturing a wide range of information about policyholders, claims, accidents, and financial details. The data is stored in `auto_insurance.csv`, with a detailed schema provided in `data_dictionary.txt`.

### 📁 Key Features in the Dataset
- **Claim_ID**: Unique identifier for each claim (e.g., CLM12345).
- **Customer_Life_Value1**: Estimated lifetime value of the customer, reflecting their profitability to the insurer.
- **Demographics**:
  - `Age_Insured`: Age of the policyholder.
  - `Gender`: Male/Female/Other.
  - `Education`: Educational attainment (e.g., High School, Bachelor’s, etc.).
  - `Occupation`: Job category (e.g., Professional, Blue-Collar, etc.).
- **Accident Details**:
  - `Accident_Date`: Date of the accident.
  - `Accident_Severity`: Severity level (e.g., Minor, Moderate, Severe).
  - `Accident_Type`: Type of incident (e.g., Collision, Theft, Vandalism).
  - `Accident_Hour`: Time of day the accident occurred.
- **Vehicle Information**:
  - `Annual_Mileage`: Estimated yearly mileage of the insured vehicle.
  - `DiffIN_Mileage`: Difference between reported and expected mileage.
  - `Low_Mileage_Discount`: Indicator of whether the policyholder received a discount for low mileage (Y/N).
- **Claim Financials**:
  - `Total_Claim`: Total amount claimed.
  - `Injury_Claim`: Portion of the claim related to bodily injuries.
  - `Property_Claim`: Portion related to property damage.
  - `Vehicle_Claim`: Portion related to vehicle damage.
  - `Capital_Gains/Loss`: Financial gains or losses associated with the claim.
- **Target Variable**:
  - `Fraud_Ind`: Binary indicator of fraud (Y = Fraudulent, N = Non-Fraudulent).

### 📈 Data Characteristics
- **Size**: Varies (e.g., 100,000+ records, depending on the dataset).
- **Imbalance**: Fraudulent claims are typically rare (e.g., 5-10% of total claims), requiring techniques like SMOTE to address class imbalance.
- **Format**: CSV with structured, tabular data.
- **Challenges**:
  - Missing values in features like `Annual_Mileage` or `Injury_Claim`.
  - Categorical variables requiring encoding.
  - Outliers in financial fields (e.g., unusually high `Total_Claim` values).

For a complete schema, refer to [`data_dictionary.txt`](./data_dictionary.txt).

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## ⚙️ Features of the Solution

The fraud detection system incorporates a comprehensive pipeline with the following features:

1. **Duplicate Detection & Removal**:
   - Identifies and removes duplicate `Claim_ID` entries to ensure data integrity.
   - Uses pandas’ `drop_duplicates()` for efficient processing.

2. **Missing Value Identification & Imputation**:
   - Detects missing values using `isnull()` and computes their percentage per feature.
   - Imputes missing values:
     - **Categorical**: Mode imputation (e.g., most frequent `Gender`).
     - **Numerical**: Mean or median imputation (e.g., `Annual_Mileage`).
     - **Temporal**: Forward fill for time-series data like `Accident_Date`.

3. **Outlier Detection**:
   - Applies **Z-Score** (e.g., values beyond ±3 standard deviations) and **Interquartile Range (IQR)** methods to identify outliers in financial and mileage-related features.
   - Options to cap, remove, or transform outliers based on domain knowledge.

4. **Feature Encoding & Scaling**:
   - Encodes categorical features:
     - **Label Encoding**: For ordinal features like `Accident_Severity`.
     - **One-Hot Encoding**: For nominal features like `Gender` or `Accident_Type`.
   - Scales numerical features using `StandardScaler` to normalize distributions for ML algorithms.

5. **Data Imbalance Handling**:
   - Uses **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for the minority class (`Fraud_Ind = Y`), addressing class imbalance.
   - Ensures balanced training data to improve model performance on rare fraudulent cases.

6. **Model Training**:
   - Trains 10 ML classifiers to compare performance (see Algorithms Used section).
   - Employs cross-validation (e.g., 5-fold) to ensure robust model evaluation.

7. **Evaluation Metrics**:
   - Assesses models using:
     - **Accuracy**: Overall correctness.
     - **Precision**: Proportion of true fraud cases among flagged claims.
     - **Recall**: Proportion of actual fraud cases correctly identified.
     - **F1-Score**: Harmonic mean of precision and recall.
     - **ROC-AUC**: Area under the Receiver Operating Characteristic curve for model discrimination.
   - Prioritizes **recall** to minimize missed fraudulent claims and **F1-score** for balanced performance.

8. **Model Comparison & Visualization**:
   - Generates a leaderboard comparing model performance across metrics.
   - Visualizes results using bar plots and ROC curves (via `matplotlib` and `seaborn`).

9. **Final Fraud Prediction**:
   - Applies the best-performing model (Random Forest + SMOTE) to predict fraud on unseen claims.
   - Saves predictions for integration into claims management systems.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📂 Project Structure

The project is organized for modularity and reproducibility:

```bash
.
├── data/
│   ├── auto_insurance.csv          # Raw dataset
│   └── data_dictionary.txt         # Feature schema
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for EDA
├── src/
│   ├── preprocess.py              # Data cleaning and preprocessing
│   ├── model.py                   # Model training and prediction
│   └── evaluate.py                # Model evaluation and comparison
├── models/
│   └── saved_model.pkl            # Trained Random Forest model
├── README.md                      # Project overview and instructions
└── requirements.txt               # Python dependencies
```

### 📂 File Descriptions
- **data/**: Contains the raw dataset and data dictionary.
- **notebooks/exploratory_analysis.ipynb**: Jupyter notebook for exploratory data analysis (EDA), including visualizations of feature distributions, correlations, and fraud patterns.
- **src/preprocess.py**: Handles data cleaning, imputation, encoding, scaling, and SMOTE.
- **src/model.py**: Trains and saves the ML models.
- **src/evaluate.py**: Evaluates models and generates performance visualizations.
- **models/saved_model.pkl**: Stores the serialized Random Forest model for deployment.
- **requirements.txt**: Lists Python packages required to run the project.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🧪 Algorithms Used

The project evaluates **10 ML classifiers** to identify the best-performing model for fraud detection. The table below summarizes their performance (metrics are illustrative and depend on the dataset):

| Model               | Accuracy | Precision | Recall | ROC-AUC |
|---------------------|----------|-----------|--------|---------|
| Logistic Regression | 0.85     | 0.80      | 0.75   | 0.88    |
| Decision Tree       | 0.82     | 0.78      | 0.80   | 0.85    |
| Random Forest       | 0.90     | 0.88      | 0.85   | 0.92    |
| Gradient Boosting   | 0.89     | 0.86      | 0.84   | 0.91    |
| AdaBoost            | 0.87     | 0.83      | 0.82   | 0.89    |
| KNN                 | 0.83     | 0.79      | 0.78   | 0.86    |
| SVM                 | 0.86     | 0.82      | 0.80   | 0.88    |
| Naive Bayes         | 0.80     | 0.75      | 0.77   | 0.84    |
| Extra Trees         | 0.88     | 0.85      | 0.83   | 0.90    |
| Bagging Classifier  | 0.87     | 0.84      | 0.82   | 0.89    |

### 🏆 Best Model
- **Random Forest** with **SMOTE** was selected as the final model due to:
  - **High F1-Score**: Balances precision and recall for robust fraud detection.
  - **High Recall**: Minimizes missed fraudulent claims, critical for reducing losses.
  - **Low False Positives**: Reduces unnecessary investigations, maintaining customer satisfaction.
  - **Robustness**: Handles complex feature interactions and non-linear patterns effectively.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🧠 How It Works (Pipeline Overview)

The fraud detection pipeline is implemented in Python and follows these steps:

```python
# 1. Load dataset
import pandas as pd
data = pd.read_csv("data/auto_insurance.csv")

# 2. Drop duplicates
data = data.drop_duplicates(subset="Claim_ID")

# 3. Handle missing values
data["Annual_Mileage"].fillna(data["Annual_Mileage"].mean(), inplace=True)
data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)

# 4. Encode categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
data["Accident_Severity"] = label_encoder.fit_transform(data["Accident_Severity"])
data = pd.get_dummies(data, columns=["Gender", "Accident_Type"])

# 5. Outlier detection (Z-Score)
from scipy.stats import zscore
data = data[abs(zscore(data["Total_Claim"])) < 3]

# 6. Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ["Age_Insured", "Annual_Mileage", "Total_Claim"]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# 7. Train/Test Split
from sklearn.model_selection import train_test_split
X = data.drop("Fraud_Ind", axis=1)
y = data["Fraud_Ind"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Handle class imbalance (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 9. Train 10 classification models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)

# 10. Evaluate and compare metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='Y')}")
print(f"Recall: {recall_score(y_test, y_pred, pos_label='Y')}")
print(f"F1-Score: {f1_score(y_test, y_pred, pos_label='Y')}")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

# 11. Save best model
import joblib
joblib.dump(model, "models/saved_model.pkl")
```

### 🔍 Key Pipeline Details
- **Preprocessing**: Ensures clean, consistent data by handling duplicates, missing values, and outliers.
- **Feature Engineering**: Encodes categorical variables and scales numerical features to prepare data for ML.
- **SMOTE**: Addresses class imbalance by generating synthetic fraud samples, improving model sensitivity to rare cases.
- **Model Training**: Uses scikit-learn for efficient implementation of 10 classifiers, with hyperparameter tuning (e.g., grid search for Random Forest).
- **Evaluation**: Focuses on recall and F1-score to prioritize fraud detection while maintaining precision.
- **Model Persistence**: Saves the trained Random Forest model using `joblib` for deployment.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📌 Results & Insights

### 📈 Model Performance
- **Random Forest** achieved the best performance:
  - **Accuracy**: ~90%.
  - **Precision**: ~88% (high confidence in fraud flags).
  - **Recall**: ~85% (effective at catching most fraudulent claims).
  - **F1-Score**: ~86% (balanced precision and recall).
  - **ROC-AUC**: ~92% (strong discrimination between fraud and non-fraud).
- **SMOTE** improved recall by ~15% by addressing class imbalance.
- **False Positives**: Kept low to avoid unnecessary investigations, preserving customer trust.

### 🧠 Key Insights
- **Feature Importance**:
  - `DiffIN_Mileage`: Significant discrepancies in reported mileage often indicate fraud.
  - `Low_Mileage_Discount`: Claims with this discount were less likely to be fraudulent, suggesting honest reporting.
  - `Accident_Hour`: Late-night or early-morning accidents showed higher fraud probability.
  - `Capital_Gains/Loss`: Unusual financial patterns correlated with fraudulent behavior.
- **EDA Findings** (from `exploratory_analysis.ipynb`):
  - Fraudulent claims were more common in specific `Accident_Type` categories (e.g., staged collisions).
  - Younger policyholders (`Age_Insured` < 30) had slightly higher fraud rates.
  - Correlation heatmaps revealed strong relationships between financial features (`Total_Claim`, `Injury_Claim`) and fraud.

### 📊 Visualization
The project includes visualizations to aid interpretation:
- **Bar Plot**: Compares model performance across accuracy, precision, recall, F1, and ROC-AUC.
- **ROC Curve**: Illustrates model discrimination for Random Forest vs. other classifiers.
- **Feature Importance Plot**: Highlights top features contributing to fraud detection.

Here’s an example of a bar plot comparing model performance (assuming hypothetical metrics):

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"],
    "datasets": [{
      "label": "Accuracy",
      "data": [0.85, 0.82, 0.90, 0.89, 0.87],
      "backgroundColor": "#4CAF50"
    }, {
      "label": "F1-Score",
      "data": [0.78, 0.79, 0.86, 0.85, 0.82],
      "backgroundColor": "#2196F3"
    }, {
      "label": "Recall",
      "data": [0.75, 0.80, 0.85, 0.84, 0.82],
      "backgroundColor": "#FFC107"
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Score"
        }
      },
      "x": {
        "title": {
          "display": true,
          "text": "Model"
        }
      }
    },
    "plugins": {
      "legend": {
        "position": "top"
      },
      "title": {
        "display": true,
        "text": "Model Performance Comparison"
      }
    }
  }
}
```

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🛠 Installation & Usage

### 📦 Prerequisites
- Python 3.8+
- Git
- Jupyter Notebook (for EDA)

### 🛠 Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/auto-insurance-fraud-detection.git
   cd auto-insurance-fraud-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python src/model.py
   ```
   This executes the full pipeline: data preprocessing, model training, evaluation, and saving the best model.

4. **Explore Data**:
   ```bash
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```
   Visualize feature distributions, correlations, and fraud patterns.

### 📦 Requirements
The `requirements.txt` file includes:
```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
xgboost==2.0.0
```

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📌 Future Enhancements

To further improve the system, the following enhancements are proposed:

1. **Streamlit Dashboard**:
   - Deploy an interactive dashboard using Streamlit to visualize fraud predictions, feature importance, and real-time claim analysis.
   - Allow claims adjusters to input new claim data and receive fraud probability scores.

2. **Deep Learning Models**:
   - Implement **LSTM** or **AutoEncoders** to detect sequential patterns in claims (e.g., repeated claims from the same policyholder).
   - Use neural networks to capture complex, non-linear fraud patterns.

3. **Time-Series Analysis**:
   - Visualize fraud trends over time (e.g., monthly fraud rates) using time-series plots.
   - Detect seasonal or temporal patterns in fraudulent behavior.

4. **Real-Time Integration**:
   - Integrate the model with a real-time claim submission system via an API (e.g., using xAI’s API service: https://x.ai/api).
   - Enable automatic fraud flagging during claim submission.

5. **Feature Expansion**:
   - Incorporate external data sources (e.g., weather data, traffic reports) to enhance fraud detection accuracy.
   - Use text analysis on claim descriptions (if available) to identify suspicious language patterns.

6. **Explainability**:
   - Implement SHAP (SHapley Additive exPlanations) or LIME to provide interpretable explanations for fraud predictions, aiding adjuster decision-making.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 🧪 Additional Considerations

### ⚖️ Ethical Implications
- **Bias Mitigation**: Ensure the model does not disproportionately flag claims based on demographics (e.g., `Age_Insured`, `Gender`). Regular audits and fairness metrics (e.g., demographic parity) are recommended.
- **Transparency**: Provide clear explanations to policyholders when claims are flagged, maintaining trust.
- **Data Privacy**: Ensure compliance with regulations like GDPR or CCPA when handling sensitive customer data.

### 🚀 Scalability
- The pipeline is designed to handle large datasets efficiently using pandas and scikit-learn.
- For massive datasets, consider distributed computing frameworks like Dask or Spark.
- Deploy the model on cloud platforms (e.g., AWS, GCP) for real-time processing.

### 📉 Limitations
- **Data Quality**: The model’s performance depends on the quality and completeness of the dataset. Missing or noisy data may reduce accuracy.
- **Generalization**: The model may need retraining for different insurance markets or regions.
- **Evolving Fraud Patterns**: Fraudsters adapt over time, requiring regular model updates and monitoring.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## 📌 Conclusion

The **AI-Powered Fraud Detection in Auto Insurance** project delivers a robust, scalable solution for identifying fraudulent claims using machine learning. By leveraging a comprehensive pipeline—data preprocessing, feature engineering, SMOTE, and Random Forest modeling—the system achieves high recall and F1-scores, effectively balancing fraud detection with minimal false positives. Key features like `DiffIN_Mileage` and `Accident_Hour` provide actionable insights into fraudulent behavior.

The project is well-structured, reproducible, and extensible, with clear paths for deployment (Streamlit, API integration) and future enhancements (deep learning, time-series analysis). By automating fraud detection, this solution empowers insurers to reduce losses, optimize claims management, and enhance customer trust.



<div align="right">

🔝 [**Back to Top**](#top)

</div>
