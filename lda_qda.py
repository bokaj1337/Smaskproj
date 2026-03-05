import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from load_data import get_ready_data

cleaned_fil = get_ready_data()
random_state = 69
X = cleaned_fil.drop(columns=["increase_stock"])
Y = cleaned_fil["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=69)

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, Y_train)
lda_pred = lda_model.predict(X_test)

lda_accuracy = accuracy_score(Y_test, lda_pred)
print(f"\nLDA Accuracy: {lda_accuracy:.4f}")

qda_model = QuadraticDiscriminantAnalysis(reg_param=0.75) # Lekte med reg_param, ändra tillbaka om ni gör nåt
qda_model.fit(X_train, Y_train)
qda_pred = qda_model.predict(X_test)

qda_accuracy = accuracy_score(Y_test, qda_pred)
print(f"\nAccuracy: {qda_accuracy:.4f}")
print(f"\nClassification Report:\n{classification_report(Y_test, qda_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(Y_test, qda_pred)}")

print("\n" + "=" * 50)
print("Model Comparison")
print("=" * 50)
print(f"LDA Accuracy: {lda_accuracy:.4f}")
print(f"QDA Accuracy: {qda_accuracy:.4f}")

# Visualize accuracies
models = ['LDA', 'QDA']
accuracies = [lda_accuracy, qda_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.ylabel('Accuracy Score')
plt.title('LDA vs QDA Accuracy Comparison')
plt.ylim([0, 1])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
