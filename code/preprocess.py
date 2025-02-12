import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data/metadata.csv')

col = ['customerID', 'Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn']

new_df = df[col]
new_df.to_csv('new_df', index = False)