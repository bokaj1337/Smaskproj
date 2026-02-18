import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import get_ready_data

cleaned_fil = get_ready_data()

X = cleaned_fil.drop(columns=["increase_stock"])
Y = cleaned_fil["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100) #Test size är på 30 % men större träningsdata kommer inte ge högre träffsäkerhet
model = LogisticRegression(penalty='l1', solver='liblinear',max_iter=1000)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

# Vi kör cross validation på ALL data (X och Y) för att se den generella prestandan
cv_scores = cross_val_score(model, X, Y, cv=skf, scoring='f1_macro')

print("--- Cross Validation Resultat (F1-Macro) ---")
print(f"Scores för varje fold: {cv_scores}")
print(f"Medelvärde: {cv_scores.mean():.4f}")
print(f"Standardavvikelse (variation): {cv_scores.std():.4f}")
print("-" * 40)



model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(Y_test, Y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")


