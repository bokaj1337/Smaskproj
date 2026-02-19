import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
''' This file contains functions to load and preprocess the data.
    Use get_ready_data() to get cleaned data that is normalized (gaussian),
    use get_unscaled_data() to get cleaned data that is not normalized.
'''
def load_raw_data():
    return pd.read_csv("training_data_VT2026.csv")

def remove_constant_data(df):
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    print(f"Dropping constant columns: {constant_columns}")
    return df.drop(constant_columns, axis=1)    

def process_data(df, scaler=StandardScaler()):
    '''
    Makes stock_demand binary and scales data.
    '''
    out = df.copy()
    label_mapping = {"low_bike_demand":1, "high_bike_demand":0}
    out["increase_stock"] = out["increase_stock"].map(label_mapping)
    numerical_columns = ["temp","dew","humidity","precip","snowdepth","windspeed","cloudcover","visibility"]
    for col in numerical_columns:
        out[col] = scaler.fit_transform(out[[col]])
    return out

def get_ready_data():
    df = load_raw_data()
    df = remove_constant_data(df)
    df = process_data(df)
    return df

def create_attribute_type_dict(df):
    attribute_type=dict([("temp",'numerical'),("dew",'numerical'),("humidity",'numerical'),
                     ("precip",'numerical'),("snowdepth",'numerical'),("windspeed",'numerical'),
                     ("cloudcover",'numerical'),("visibility",'numerical')])
    attribute_type.update(dict([("hour_of_day",'categorical'),("day_of_week",'categorical'),
                                ("month",'categorical')]))
    attribute_type.update(dict([("holiday",'binary'),("weekday",'binary'),("summertime",'binary')]))
    attribute_type.update(dict([("increase_stock",'target')]))
    return attribute_type

def get_unscaled_data():
    df = load_raw_data()
    df = remove_constant_data(df)
    label_mapping = {"low_bike_demand":0, "high_bike_demand":1}
    df["increase_stock"] = df["increase_stock"].map(label_mapping)
    return df