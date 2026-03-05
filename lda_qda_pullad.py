import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from load_data import get_ready_data

cleaned_fil = get_ready_data()

X = cleaned_fil.drop(columns=["increase_stock"])
Y = cleaned_fil["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=69)

lda_params = {
    'solver': ['lsqr', 'eigen'],
    'shrinkage': [None, 'auto', 0.0, 0.25, 0.5, 0.75, 1.0]
}
lda_grid = GridSearchCV(LinearDiscriminantAnalysis(), lda_params, cv=5, scoring='f1', error_score=0)
lda_grid.fit(X_train, Y_train)
print(f"Best LDA Parameters: {lda_grid.best_params_}")
print(f"Best LDA Cross-Validation Score: {lda_grid.best_score_:.4f}")
lda_mean_test = lda_grid.cv_results_['mean_test_score'].mean()
lda_std_test = lda_grid.cv_results_['std_test_score'].mean()
print(f"LDA Mean Test Score: {lda_mean_test:.4f} ± {lda_std_test:.4f}")

lda_model = lda_grid.best_estimator_
lda_pred = lda_model.predict(X_test)
lda_accuracy = accuracy_score(Y_test, lda_pred)
lda_f1 = f1_score(Y_test, lda_pred)
print(f"\nLDA Accuracy on Test Set: {lda_accuracy:.4f}")
print(f"LDA F1-Score on Test Set: {lda_f1:.4f}\n")

qda_params = {
    'reg_param': [0.1, 0.25, 0.5, 0.75, 1.0]
}
qda_grid = GridSearchCV(QuadraticDiscriminantAnalysis(), qda_params, cv=5, scoring='f1', error_score=0)
qda_grid.fit(X_train, Y_train)
print(f"Best QDA Parameters: {qda_grid.best_params_}")
print(f"Best QDA Cross-Validation Score: {qda_grid.best_score_:.4f}")
qda_mean_test = qda_grid.cv_results_['mean_test_score'].mean()
qda_std_test = qda_grid.cv_results_['std_test_score'].mean()
print(f"QDA Mean Test Score: {qda_mean_test:.4f} ± {qda_std_test:.4f}")

qda_model = qda_grid.best_estimator_
qda_pred = qda_model.predict(X_test)
qda_accuracy = accuracy_score(Y_test, qda_pred)
qda_f1 = f1_score(Y_test, qda_pred)
print(f"\nQDA Accuracy on Test Set: {qda_accuracy:.4f}")
print(f"QDA F1-Score on Test Set: {qda_f1:.4f}")
#print(f"\nClassification Report:\n{classification_report(Y_test, qda_pred)}")
#print(f"\nConfusion Matrix:\n{confusion_matrix(Y_test, qda_pred)}")