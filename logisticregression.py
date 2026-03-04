import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import get_ready_data

cleaned_fil = get_ready_data()

X = cleaned_fil.drop(columns=["increase_stock", "holiday", "month"])
Y = cleaned_fil["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=69)
model = LogisticRegression(solver='liblinear',max_iter=1000) 

param_grid = {'C': [0.1, 1 ,5,10,100], 'penalty': ['l1','l2'], 'class_weight': [{0:1,1:1},{0:1,1:2},'balanced'] }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='f1_macro', verbose=1)

grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)

best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)

print(f"Accuracy på testdata: {accuracy_score(Y_test, Y_pred):.4f}")
print(f"Classification Report:{classification_report(Y_test, Y_pred)}")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pred)}")

