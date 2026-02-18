import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from data_analysis import corr_mat
from load_data import get_unscaled_data, create_attribute_type_dict
from data_analysis import chi2_stats, corr_mat

original_data = get_unscaled_data()
attribute_type = create_attribute_type_dict(original_data)
extended_data = original_data.copy()

def add_app_temp_features(df):
    # Add vapor pressure as feature
    vapor_pressure = 6.105/100*df['humidity']*np.exp(17.27*df['temp']/(df['temp']+237.3)) # from wiki -apparent temperature formula
    extended_data['vapor_pressure'] = vapor_pressure
    attribute_type['vapor_pressure'] = 'numerical'

    # add apparent temperature as feature
    apparent_temp = df['temp'] + 0.33*vapor_pressure - 0.7*df['windspeed'] - 4 # from wiki -apparent temperature formula
    extended_data['apparent_temp'] = apparent_temp
    attribute_type['apparent_temp'] = 'numerical'
    return df
extended_data = add_app_temp_features(extended_data)
#corr_mat(extended_data, attribute_type)

def test_transformed_features(df, feature, att_dict):
    transformations = {"sin":np.sin, "cos":np.cos, "tan":np.tan, "log":np.log, "sqrt":np.sqrt,
                           "exp":np.exp, "reciprocal":lambda x: 1/x, "square":np.square,"cube":lambda x: x**3}
    transformed = df[feature]
    uppdated_att_dict = att_dict.copy()
    for name, transform in transformations.items():
        transformed[name]=transform(df[feature])
        uppdated_att_dict[f"{name}_{feature}"] = 'numerical'
    corr_mat(transformed, uppdated_att_dict)

test_transformed_features(extended_data, "temp", attribute_type)