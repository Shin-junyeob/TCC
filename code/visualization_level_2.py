import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
file_path = "../data/new_metadata.csv"  # 파일 경로를 상황에 맞게 수정하세요.
df = pd.read_csv(file_path)

# 2. Churn이 "Yes"인 고객만 필터링
df_churn_yes = df[df["Churn"] == "Yes"]
df_churn_no = df[df["Churn"] == 'No']

# 3. 분석에 사용할 변수 선택
selected_features = [
    'Partner', 'Dependents', 'InternetService', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TenureGroup', 'PlanChange'
]

# 4. 선택한 변수들만 추출
df_selected_yes = df_churn_yes[selected_features].copy()
df_selected_no = df_churn_no[selected_features].copy()

# 5. 범주형 변수에 대해 One-Hot Encoding 적용
#    (MonthlyCharges는 연속형 변수이므로 그대로 유지)
categorical_features = ['Partner', 'Dependents', 'InternetService', 'Contract',
                        'PaperlessBilling', 'PaymentMethod', 'TenureGroup', 'PlanChange']
df_encoded_yes = pd.get_dummies(df_selected_yes, columns=categorical_features, drop_first=False)
df_encoded_no = pd.get_dummies(df_selected_no, columns=categorical_features, drop_first=False)


# 6. 상관관계 행렬 계산
corr_matrix_yes = df_encoded_yes.corr()
corr_matrix_no = df_encoded_no.corr()

# 7. 음수 상관관계 계수를 모두 0으로 변경
corr_matrix_yes[corr_matrix_yes < 0] = 0
corr_matrix_no[corr_matrix_no < 0] = 0

# 8-1. Yes상관관계 히트맵 시각화
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix_yes, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap for Selected Features (Churn = Yes) - Negative set to 0")

plt.savefig("../results/level_2/correlation_heatmap_yes.png", dpi=300, bbox_inches="tight")

# 8-2. No상관관계 히트맵 시각화
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix_no, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap for Selected Features (Churn = No) - Negative set to 0")

plt.savefig("../results/level_2/correlation_heatmap_no.png", dpi=300, bbox_inches="tight")