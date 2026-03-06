import numpy as np
import pandas as pd
from load_data import get_ready_data
from knn import best_model as knn_model


#cleaned_fil = get_ready_data()

#predicted = knn_model.predict(cleaned_fil)


# print(knn_model.predict_proba(cleaned_fil))

# with open("predictions.csv", "w") as f:
#     f.write(",".join(map(str, predicted)))

final_test_data = get_ready_data(file_path="test_data_VT2026.csv")
#print(final_test_data)

predicted = knn_model.predict(final_test_data)
print(predicted)
with open("predictions.csv", "w") as f:
    f.write(",".join(map(str, predicted)))
