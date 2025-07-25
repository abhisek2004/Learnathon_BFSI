# Learnathon_BFSI

Here is a **detailed and professional README.md file** for your project titled:

---

# 🚗 AI-Powered Fraud Detection in Auto Insurance

### 🧠 Predictive Modeling for Smarter Claims Management

---

## 📌 Problem Statement

Insurance fraud in the auto sector leads to billions in losses every year, resulting in higher premiums for customers and significant financial strain on insurance companies. The goal of this project is to leverage AI/ML techniques to build a robust **fraud detection system** that can accurately flag potentially fraudulent claims using structured historical insurance claim data.

This solution helps insurers:

* Automate fraud detection in real-time.
* Prioritize claims that require deeper investigation.
* Enhance the efficiency of the claims management pipeline.

---

## 📊 Dataset Description

The dataset contains **structured claim records** with 50+ features ranging from policy details, customer demographics, accident specifics, and financial information.

📁 Data Dictionary Highlights:

* `Claim_ID`: Unique claim identifier
* `Customer_Life_Value1`: Estimated value of customer
* `Age_Insured`, `Gender`, `Education`, `Occupation`, etc.
* `Accident_Date`, `Accident_Severity`, `Accident_Type`
* `Annual_Mileage`, `DiffIN_Mileage`, `Low_Mileage_Discount`
* `Total_Claim`, `Injury_Claim`, `Property_Claim`, `Vehicle_Claim`
* `Fraud_Ind`: **Target column** (Y/N)

For full schema, see [`data_dictionary.txt`](./data_dictionary.txt)

---

## ⚙️ Features of the Solution

* ✅ **Duplicate detection & removal**
* 🧩 **Missing value identification & imputation**
* 🚨 **Outlier detection using Z-Score and IQR methods**
* 🧠 **Feature encoding & scaling**
* 🔍 **Data imbalance handling** using SMOTE
* ⚒️ **Model training with 10 ML classifiers**
* 📈 **Evaluation using accuracy, precision, recall, F1, ROC-AUC**
* 🏆 **Model comparison & leaderboard visualization**
* 🎯 Final fraud prediction on unseen claims

---

## 📂 Project Structure

```bash
.
├── data/
│   ├── auto_insurance.csv
│   └── data_dictionary.txt
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── models/
│   └── saved_model.pkl
├── README.md
└── requirements.txt
```

---

## 🧪 Algorithms Used

We evaluated 10 models:

| Model               | Accuracy | Precision | Recall | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------- |
| Logistic Regression | ✅        | ✅         | ✅      | ✅       |
| Decision Tree       | ✅        | ✅         | ✅      | ✅       |
| Random Forest       | ✅        | ✅         | ✅      | ✅       |
| Gradient Boosting   | ✅        | ✅         | ✅      | ✅       |
| AdaBoost            | ✅        | ✅         | ✅      | ✅       |
| KNN                 | ✅        | ✅         | ✅      | ✅       |
| SVM                 | ✅        | ✅         | ✅      | ✅       |
| Naive Bayes         | ✅        | ✅         | ✅      | ✅       |
| Extra Trees         | ✅        | ✅         | ✅      | ✅       |
| Bagging Classifier  | ✅        | ✅         | ✅      | ✅       |

> ✅ Final model: **Random Forest** + **SMOTE** for best F1 & recall.

---

## 🧠 How It Works (Pipeline Overview)

```python
1. Load dataset
2. Drop duplicates
3. Handle missing values (mode, mean, or forward fill)
4. Encode categorical features (Label/One-Hot Encoding)
5. Outlier detection (Z-Score method)
6. Feature scaling (StandardScaler)
7. Train/Test Split
8. Handle class imbalance (SMOTE)
9. Train 10 classification models
10. Evaluate and compare metrics
11. Save best model
```

---

## 📌 Results & Insights

* Features like `DiffIN_Mileage`, `Low_Mileage_Discount`, `Accident_Hour`, and `Capital_Gains/Loss` showed strong correlation with fraudulent behavior.
* Data imbalance was significantly handled using **SMOTE**, improving recall by \~15%.
* **Random Forest** showed the best performance with high F1-score and low false positives.

---

## 🛠 Installation & Usage

```bash
git clone https://github.com/your-username/auto-insurance-fraud-detection.git
cd auto-insurance-fraud-detection
pip install -r requirements.txt
```

Run the pipeline:

```bash
python src/model.py
```

For detailed data exploration:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

---

## 📌 Future Enhancements

* ✅ Deploy as Streamlit dashboard
* 🔍 Use deep learning (LSTM/AutoEncoders) for sequential claim fraud
* 📉 Time-series fraud trend visualization
* 🛰 Integration with real-time claim submission system

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 👨‍💻 Developed By

* **Abhisek Panda** – [GitHub](https://github.com/abhisek2004)
* Team: *  *
