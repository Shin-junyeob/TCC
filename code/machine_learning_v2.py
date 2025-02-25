import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from scipy.stats import randint, uniform

# 1. 데이터 로드
file_path = "../data/new_metadata.csv"
df = pd.read_csv(file_path)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
y = df["Churn"]
X = df.drop(columns=["Churn"])

# 범주형 변수 처리
X = pd.get_dummies(X, drop_first=True)

# 2. 데이터 분할 (train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. SMOTEENN 적용: 학습 데이터에만
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

# 4. 모델 및 파라미터 설정 (탐색 범위 확장 예시)
models = [
    ('Random Forest', RandomForestClassifier(random_state=42),
        {'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30]}),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1]}),
    ('Support Vector Machine', SVC(random_state=42, class_weight='balanced', probability=True),
        {'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']}),
    ('Logistic Regression', LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
        {'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']}),
    ('K-Nearest Neighbors', KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']}),
    ('Decision Tree', DecisionTreeClassifier(random_state=42),
        {'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]}),
    ('XG Boost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        {'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 500),
        'subsample': uniform(0.7, 0.3)}),
    ('Naive Bayes', GaussianNB(), {})
]

model_scores = []
best_model = None
best_accuracy = 0.0

# 5. 각 모델에 대해 교차 검증(cv=5)으로 튜닝 및 평가
for name, model, param_grid in models:
    print("-----", name, "-----")
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', model)
    ])

    # XG Boost는 RandomizedSearchCV
    if name == 'XG Boost' and param_grid:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={'model__' + k: v for k, v in param_grid.items()},
            n_iter=50,  # 적절히 조정
            cv=5,       # 교차 검증 폴드 수
            verbose=1, 
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train_res, y_train_res)
        pipeline = search.best_estimator_
    elif param_grid:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={'model__' + k: v for k, v in param_grid.items()},
            cv=5,       # 교차 검증 폴드 수
            verbose=1,
            n_jobs=-1
        )
        search.fit(X_train_res, y_train_res)
        pipeline = search.best_estimator_
    
    pipeline.fit(X_train_res, y_train_res)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_scores.append({'Model': name, 'Accuracy': accuracy})

    print("Test Accuracy:", round(accuracy, 4))
    print(classification_report(y_test, y_pred))
    print()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

scores_df = pd.DataFrame(model_scores)
print("Model Scores:")
print(scores_df)
print("Best Model:", best_model)
print("Best Accuracy:", best_accuracy)

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=scores_df, palette="pastel")
plt.ylim(0, 1)
for i, v in enumerate(scores_df["Accuracy"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
plt.title("Model Accuracy Comparison (SMOTEENN, cv=5)")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../results/model_accuracy_comparison_v2.png", dpi=300, bbox_inches="tight")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Best Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("../results/confusion_matrix_best_model.png", dpi=300, bbox_inches="tight")