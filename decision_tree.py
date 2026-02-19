import numpy as np
import pandas as pd
from load_data import get_ready_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


rf_orignal_data = RandomForestClassifier(random_state=69, oob_score=True)
param_grid = {
    "n_estimators": [100, 200, 500],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8],
    'max_depth': [5, 10, 20, None],
}

grid_search_rf = GridSearchCV(estimator=rf_orignal_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

if __name__ == "__main__":
    data = get_ready_data()
    X = data.drop("increase_stock", axis=1)
    y = data["increase_stock"]
    grid_search_rf.fit(X, y)
    print("Best parameters found: ", grid_search_rf.best_params_)