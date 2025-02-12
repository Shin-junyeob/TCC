import matplotlib.pyplot as plt
import seaborn as sns
from prepare import *

# 1) 부양가족 여부에 따른 이탈여부 Dependents - Churn
class results1:
    def __init__(self):
        results1 = df.groupby('Dependents')['Churn'].mean()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=['No', 'Yes'], y=results1.values, palette='Set2')
        plt.title('Churn Rate by Dependents')
        plt.xlabel('Dependents')
        plt.ylabel('Churn Rate')
        plt.savefig('../results/results1.png')
        plt.show()

# 2) 인터넷 서비스 사용 여부에 따른 이탈여부 InternetService - Churn
class results2:
    def __init__(self):
        results2 = df.groupby('InternetService')['Churn'].mean()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=['DSL', 'Fiber optic', 'No'], y=results2.values, palette='Set2')
        plt.title('Churn Rate by InternetService')
        plt.xlabel('InternetService')
        plt.ylabel('Churn Rate')
        plt.savefig('../results/results2.png')
        plt.show()
# 3) 사용기간 별 이탈 여부 tenure - Churn
class results3:
    def __init__(self):
        tenure_threshold = df.tenure.median()
        charge_threshold = df.MonthlyCharges.median()

        conditions = [
            (df['tenure'] < tenure_threshold) & (df['MonthlyCharges'] < charge_threshold),
            (df['tenure'] < tenure_threshold) & (df['MonthlyCharges'] >= charge_threshold),
            (df['tenure'] >= tenure_threshold) & (df['MonthlyCharges'] < charge_threshold),
            (df['tenure'] >= tenure_threshold) & (df['MonthlyCharges'] >= charge_threshold)
        ]

        labels = ['Short Tenure, Low Charge', 'Short Tenure, High Charge', 'Long Tenure, Low Charge', 'Long Tenure, High Charge']

        df['Tenure_Charge_Group'] = pd.cut(df['tenure'], bins=[-1, tenure_threshold, float('inf')], labels=['Short Tenure', 'Long Tenure'])
        df['Charge_Group'] = pd.cut(df['MonthlyCharges'], bins=[-1, charge_threshold, float('inf')], labels=['Low Charge', 'High Charge'])

        churn_rate = df.groupby(['Tenure_Charge_Group', 'Charge_Group'])['Churn'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Tenure_Charge_Group', y='Churn', hue='Charge_Group', data=churn_rate, palette='Set2')
        plt.title('Churn Rate by Tenure and Monthly Charges')
        plt.xlabel('Tenure and Charge Groups')
        plt.ylabel('Churn Rate')
        plt.legend(title='Charge Group')
        plt.savefig('../results/results3.png')
        plt.show()

results3()