import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_data import get_ready_data

cleaned_fil = get_ready_data()

X = cleaned_fil.drop(columns=["increase_stock"])
Y = cleaned_fil["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42) #Test size är på 30 % men större träningsdata kommer inte ge högre träffsäkerhet
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

