import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('../data/metadata.csv')

col = ['Partner', 'Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

df = df[col]
new_df = df.copy()

new_df['tenure'] = pd.to_numeric(new_df['tenure'], errors='coerce')
new_df['TotalCharges'] = pd.to_numeric(new_df['TotalCharges'], errors='coerce')
new_df['MonthlyCharges'] = pd.to_numeric(new_df['MonthlyCharges'], errors='coerce')

mask = new_df['TotalCharges'].isna() & (new_df['tenure'] == 0)
new_df.loc[mask, 'TotalCharges'] = new_df.loc[mask, 'MonthlyCharges']

new_df['TenureGroup'] = new_df.apply(lambda x: '<1-year' if x['tenure'] < 12 else ('1-year≤ & ≤3-year' if 12 <= x['tenure'] < 36 else '3-year>'), axis = 1)

new_df['PlanChange'] = new_df.apply(lambda row: 'Upgrade' if row['TotalCharges'] > (row['tenure'] * row['MonthlyCharges'])
                                    else ('Downgrade' if row['TotalCharges'] < (row['tenure'] * row['MonthlyCharges'])
                                        else 'Same'), axis = 1)
new_df.to_csv('../data/new_metadata.csv', index = False)

print('new_metadata.csv 저장 완료.')