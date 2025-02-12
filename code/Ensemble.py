from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from prepare import *
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

pos_class = sum(y_train_resampled == 1)
neg_class = sum(y_train_resampled == 0)
ratio = pos_class / neg_class

rf = RandomForestClassifier(n_estimators = 100, max_depth = 5)
lr = LogisticRegression()
xgb = xgb.XGBClassifier(
    objective = 'binary:logistic',
    eval_metric = 'logloss',
    colsample_bytree = 0.8,
    gamma = 0,
    learning_rate = 0.2,
    max_depth = 3,
    min_child_weight = 5,
    n_estimators = 50,
    subsample = 0.9,
    scale_pos_weight = ratio
)
ensemble_model = VotingClassifier(
    estimators = [
    ('xgb', xgb),
    ('rf', rf),
    ('lr', lr)
], voting = 'hard')

ensemble_model.fit(X_train_resampled, y_train_resampled)

print(f'Ensemble Model Accuracy: {ensemble_model.score(X_test, y_test):.4f}')