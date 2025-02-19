import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load and prepare data
df = pd.read_csv('churn prediction.csv')

# Preprocess data
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['AverageMonthlySpend'] = df['MonthlyCharges']

# Drop unnecessary columns and handle missing values
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
df = df.fillna(0)

# Print unique values in Churn column to debug
print("Unique values in Churn column:", df['Churn'].unique())

# Prepare features and target - adjusted for possible different formats
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0, 1: 1, 0: 0})

# Verify target distribution
print("\nTarget distribution:")
print(y.value_counts())

# Convert categorical variables
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'AverageMonthlySpend']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train and save models
models = {
    'logistic_regression.pkl': LogisticRegression(random_state=42),
    'random_forest.pkl': RandomForestClassifier(random_state=42),
    'xgboost.pkl': xgb.XGBClassifier(random_state=42)
}

# Train and save each model
for filename, model in models.items():
    model.fit(X_train, y_train)
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model, f)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models trained and saved successfully!") 