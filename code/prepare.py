import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data/new_metadata.csv')

df = df.drop(columns=['customerID'])
features = df.columns.tolist()
target = features.pop()

encoder = LabelEncoder()
for feature in features:
    if feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        df[feature] = pd.to_numeric(df[feature], errors = 'coerce')
        df[feature] = df[feature].fillna(df[feature].mean())
    else:
        df[feature] = encoder.fit_transform(df[feature])
df['Churn'] = encoder.fit_transform(df['Churn'])


X = df[features]; y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)