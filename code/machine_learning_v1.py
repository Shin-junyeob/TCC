import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 모델들 임포트
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# 1. 데이터 로드
file_path = "../data/new_metadata.csv"
df = pd.read_csv(file_path)

# 2. 타겟 변수 및 독립 변수 설정
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
y = df["Churn"]
X = df.drop(columns=["Churn"])

# 3. 범주형 변수 처리: pd.get_dummies (drop_first 옵션은 필요에 따라 조정)
X = pd.get_dummies(X, drop_first=True)

# 4. 데이터 스케일링: StandardScaler 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. train_test_split: 80:20 비율, stratify 옵션으로 클래스 비율 유지
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 데이터 불균형 처리: SMOTE 적용
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 7. 사용할 모델들 정의 (기본 하이퍼파라미터, 추후 튜닝 예정)
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "NaiveBayes": GaussianNB()
}

# 8. 각 모델 학습 및 평가
results = {}
for name, model in models.items():
    print(f"----- {name} -----")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    results[name] = acc

print("모델별 Accuracy Scores:")
print(results)

# 9. 모델 Accuracy를 barplot으로 시각화 및 저장
plt.figure(figsize=(10, 6))
# 결과 딕셔너리를 DataFrame으로 변환
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="pastel")
plt.ylim(0, 1)
for i, v in enumerate(results_df["Accuracy"]):
    plt.text(i, v + 0.01, f"{v:.2f}%", ha="center", va="bottom")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../results/model_accuracy_comparison_v1.png", dpi=300, bbox_inches="tight")
plt.show()