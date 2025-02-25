import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = "../data/new_metadata.csv"
df = pd.read_csv(file_path)

# 2. 분석에 사용할 변수 선택
selected_features = [
    'Partner', 'Dependents', 'InternetService', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TenureGroup', 'PlanChange'
]

# 3. 각 변수별로 Churn = Yes/No 분포 시각화
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    
    # 연속형 변수는 boxplot, 범주형 변수는 countplot으로 시각화
    if feature == "MonthlyCharges":
        sns.boxplot(x="Churn", y=feature, data=df)
        plt.title(f"Boxplot of {feature} by Churn")
    else:
        sns.countplot(x=feature, hue="Churn", data=df)
        plt.title(f"Distribution of {feature} by Churn")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'../results/level_1/Distribution_of_{feature}_by_Churn', dpi=300, bbox_inches='tight')
