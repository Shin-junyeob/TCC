import pandas as pd
import ace_tools_open as tools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../data/new_metadata.csv')

# InternetService별 이탈률
rate_for_Internetservice = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100

tools.display_dataframe_to_user(name='Churn Rate by InternetService', dataframe=rate_for_Internetservice)
print()

# Contract별 이탈률
rate_for_Contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100

tools.display_dataframe_to_user(name='Churn Rate by Contract', dataframe=rate_for_Contract)
print()

# PlanChange별 이탈률률
rate_for_Planchange = df.groupby('PlanChange')['Churn'].value_counts(normalize=True).unstack() * 100

tools.display_dataframe_to_user(name='Churn Rate by PlanChange', dataframe=rate_for_Planchange)
print()

# PaymentMethod별 이탈률률
rate_for_PaymentMethod = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack() * 100

tools.display_dataframe_to_user(name='Churn Rate by PaymentMethod', dataframe=rate_for_PaymentMethod)
print()

# Tenure별 이탈률률
rate_for_Tenure = df.groupby('TenureGroup')['Churn'].value_counts(normalize=True).unstack() * 100

tools.display_dataframe_to_user(name='Churn Rate by Tenure', dataframe=rate_for_Tenure)
print()

# Partner & Dependents별 이탈률률
rate_for_Partner_Dependents = df.groupby(['Partner', 'Dependents'])['Churn'].value_counts(normalize=True).unstack() * 100

tools.display_dataframe_to_user(name='Churn Rate by Partner & Dependents', dataframe=rate_for_Partner_Dependents)
print()

# 데이터프레임 리스트
df_list = [
    ("Churn Rate by InternetService", rate_for_Internetservice),
    ("Churn Rate by Contract", rate_for_Contract),
    ("Churn Rate by PlanChange", rate_for_Planchange),
    ("Churn Rate by PaymentMethod", rate_for_PaymentMethod),
    ("Churn Rate by tenure", rate_for_Tenure),
    ("Churn Rate by Partner & Dependents", rate_for_Partner_Dependents)
]

# 하나의 TXT 파일로 저장
output_file = "../results/analysis.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for title, df_category in df_list:
        f.write(f"==== {title} ====\n")  # 섹션 제목 추가
        f.write(df_category.to_string())  # DataFrame을 문자열로 변환 후 추가
        f.write("\n\n" + "="*50 + "\n\n")  # 섹션 구분선 추가

print(f"이탈률 분석 결과가 {output_file}에 저장되었습니다!")