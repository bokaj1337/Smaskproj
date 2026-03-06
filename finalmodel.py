import numpy as np
import pandas as pd
from load_data_real import get_ready_data
from knn import best_model as knn_model


cleaned_fil = get_ready_data()

predicted = knn_model.predict(cleaned_fil)

# with open("predictions.csv", "w") as f:
#     f.write(",".join(map(str, predicted)))

