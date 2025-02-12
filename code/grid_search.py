from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from prepare import *

param_grid = {
    'max_depth' : [3, 5, 7],
    'learning_rate' : [0.01, 0.1, 0.2] ,
    'n_estimators' : [50, 100, 200],
    'subsample' : [0.7, 0.8, 0.9],
    'colsample_bytree' : [0.7, 0.8, 0.9],
    'gamma' : [0, 0.1, 0.2],
    'min_child_weight' : [1, 5, 10]
}

grid_search = GridSearchCV(estimator = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'logloss'), param_grid = param_grid, cv = 3, verbose = 1, n_jobs = -1)

grid_search.fit(X_train, y_train)

print('Best parameters found: ', grid_search.best_params_)
print(f'Best accuracy found: {grid_search.best_score_:.4f}')