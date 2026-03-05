import numpy as np
import pandas as pd
from load_data import get_ready_data
from feature_creation import get_ready_extended_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


random_state = 69
rf_orignal_data = RandomForestClassifier(random_state=random_state, class_weight='balanced_subsample') # Balanced subsample
param_grid = {
    "n_estimators": [100, 200, 500],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8],
    'max_depth': [5, 10, 15, 20, None],
}

grid_search_rf = GridSearchCV(estimator=rf_orignal_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')

if __name__ == "__main__":
    data = get_ready_extended_data()
    X = data.drop("increase_stock", axis=1)
    y = data["increase_stock"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    grid_search_rf.fit(X_train, y_train)
    print("Best parameters found: ", grid_search_rf.best_params_)

    results = pd.DataFrame(grid_search_rf.cv_results_)

    top_5 = results.sort_values(by='mean_test_score', ascending=False).head(5)
    print(top_5[['param_n_estimators', 'param_max_features', 'param_max_depth', 'mean_test_score', 'std_test_score']])


    # Test the best one on the test set
    best_model = grid_search_rf.best_estimator_
    y_pred = best_model.predict(X_test) 

    importances = best_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.sort_values(ascending=False).plot(kind='barh')
    plt.title("What features are most important for predicting 'increase_stock'?")
    plt.savefig("figures/decision_tree/feature_importance.png", bbox_inches='tight',)
    plt.show()

    # Don't optimize according to these...
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Random Forest")
    plt.savefig("figures/decision_tree/confusion_matrix.png", bbox_inches='tight',)
    plt.show()
    