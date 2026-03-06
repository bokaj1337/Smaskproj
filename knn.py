import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from load_data import get_ready_data

# Load data
cleaned_fil = get_ready_data()
X = cleaned_fil.drop(columns=["increase_stock"])
Y = cleaned_fil["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=69)

pipeline = Pipeline([
    #('scaler', StandardScaler()), det här ingår typ redan i get_ready_data() så jag tog bort det
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': range(1, 31),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=skf,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)

print("Best Parameters:")
print(grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print("-" * 40)
# Evaluate on test data
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
best_index = grid_search.best_index_
best_std = grid_search.cv_results_['std_test_score'][best_index]

print("Best CV Mean:", grid_search.best_score_)
print("Best CV Std:", best_std)
print(f"Test Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(Y_test, Y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
print(best_model)
