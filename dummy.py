import numpy as np
from load_data import get_ready_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier

random_state = 69
X = get_ready_data().drop(columns=["increase_stock"])
Y = get_ready_data()["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state)
dumb = DummyClassifier(strategy="most_frequent").fit(X_train, Y_train)
Y_pred = dumb.predict(X_test)

print(f"Accuracy på training data: {accuracy_score(Y_train, dumb.predict(X_train)):.4f}")

print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
