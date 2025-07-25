import pandas as pd
import numpy as np

# Read the original dataset
df = pd.read_csv(r"Auto_Insurance_Fraud_Claims_File03.csv")

# Convert date columns to datetime
date_cols = ['Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date',
             'Accident_Date', 'Claims_Date', 'DL_Expiry_Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Ensure numeric types for division
df['Policy_Premium'] = pd.to_numeric(df['Policy_Premium'], errors='coerce')
df['Policy_BI'] = pd.to_numeric(df['Policy_BI'], errors='coerce')

# --------------------
# Policy-related features
# --------------------
df['Days_Between_Policy_Start_and_Claim'] = (df['Claims_Date'] - df['Policy_Start_Date']).dt.days
df['Days_To_Policy_Expiry'] = (df['Policy_Expiry_Date'] - df['Claims_Date']).dt.days
df['Policy_Duration'] = (df['Policy_Expiry_Date'] - df['Policy_Start_Date']).dt.days
df['Policy_Premium_to_Coverage_Ratio'] = df['Policy_Premium'] / df['Policy_BI'].replace(0, np.nan)
df['High_Deductible_Flag'] = (df['Policy_Ded'] > df['Policy_Ded'].median()).astype(int)

# --------------------
# Accident-related features
# --------------------
df['Days_Between_Accident_and_Claim'] = (df['Claims_Date'] - df['Accident_Date']).dt.days
df['Weekend_Accident'] = df['Accident_Date'].dt.dayofweek.isin([5, 6]).astype(int)
df['Night_Accident'] = df['Accident_Hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
df['Accident_To_Policy_Start_Gap'] = (df['Accident_Date'] - df['Policy_Start_Date']).dt.days

# --------------------
# Vehicle-related features
# --------------------
df['Vehicle_Age'] = df['Accident_Date'].dt.year - df['Auto_Year']
df['High_Mileage_Flag'] = (df['Annual_Mileage'] > df['Annual_Mileage'].median()).astype(int)
df['Mileage_Change'] = df['DiffIN_Mileage'] / df['Annual_Mileage'].replace(0, np.nan)

# --------------------
# Claim behavior
# --------------------
df['Claim_Amount_to_Vehicle_Cost_Ratio'] = df['Total_Claim'] / df['Vehicle_Cost'].replace(0, np.nan)
df['Claim_Severity_Index'] = df[['Injury_Claim', 'Property_Claim', 'Vehicle_Claim']].sum(axis=1)
df['Multiple_Claims_Flag'] = df.duplicated(subset=['Policy_Num'], keep=False).astype(int)

# --------------------
# Location & mismatch indicators
# --------------------
df['Out_of_State_Claim'] = (df['Acccident_State'] != df['Policy_State']).astype(int)
df['Garage_Location_Mismatch'] = (df['Garage_Location'] != df['Acccident_City']).astype(int)

# Fill NaN values
df = df.fillna(0)

# Save the processed dataset
df.to_csv("dataset3.csv", index=False)
print("Processed dataset saved as dataset3.csv")