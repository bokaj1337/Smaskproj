import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from data_analysis import corr_mat
from load_data import get_unscaled_data, create_attribute_type_dict
from data_analysis import chi2_stats, corr_mat

original_data = get_unscaled_data()
attribute_type = create_attribute_type_dict()
extended_data = original_data.copy()

def add_app_temp_features(df, att_dict):
    # Add vapor pressure as feature
    vapor_pressure = 6.105/100*df['humidity']*np.exp(17.27*df['temp']/(df['temp']+237.3)) # from wiki -apparent temperature formula
    df['vapor_pressure'] = vapor_pressure
    att_dict['vapor_pressure'] = 'numerical'

    # add apparent temperature as feature
    apparent_temp = df['temp'] + 0.33*df['vapor_pressure'] - 0.7*df['windspeed'] - 4 # from wiki -apparent temperature formula
    df['apparent_temp'] = apparent_temp
    att_dict['apparent_temp'] = 'numerical'
    return df, att_dict

def test_transformed_features(df, feature, att_dict):
    ''' Apply various transformations to a feature and check correlation matrix.
    '''
    transformations = {"sin":np.sin, "cos":np.cos, "tan":np.tan,"atan":np.arctan, "log":np.log, "sqrt":np.sqrt,
                           "exp":np.exp, "reciprocal":lambda x: 1/x, "square":np.square,"cube":lambda x: x**3}
    transformed = df.copy()
    uppdated_att_dict = att_dict.copy()
    for name, transform in transformations.items():
        transformed[name+"_"+feature]=transform(df[feature])
        uppdated_att_dict[name+"_"+feature] = 'numerical'
    corr_mat(transformed, uppdated_att_dict)
#test_transformed_features(extended_data, "precip", attribute_type)

def add_good_weather_feature(df, att_dict):
    ''' Add a feature that is 1 if the weather conditions are good for biking, 0 otherwise.
    We define good weather as temp > 15, precip < 0.1, snowdepth < 0.1, windspeed < 10, cloudcover < 5.
    '''
    df['is_good_weather'] = ((df['temp'] > 15) & (df['precip'] < 0.1) & (df['snowdepth'] < 0.1) &
                          (df['windspeed'] < 10) & (df['cloudcover'] < 5)).astype(int)
    att_dict['is_good_weather'] = 'binary'    
    return df, att_dict

def add_is_raining_feature(df, att_dict):
    ''' Add a feature that is 1 if it is raining, 0 otherwise. We define raining as precip > 0.1.
    '''
    df['is_raining'] = (df['precip'] > 0.1).astype(int)
    att_dict['is_raining'] = 'binary'    
    return df, att_dict

def add_rush_hour_feature(df, att_dict):
    ''' Add a feature that is 1 if it is rush hour, 0 otherwise. Def rush hour as 4pm-6pm.
    '''
    df['is_rush_hour'] = ((df['hour_of_day'] >= 16) & (df['hour_of_day'] <= 18)).astype(int)
    att_dict['is_rush_hour'] = 'binary'    
    return df, att_dict

def add_dew_point_depression_feature(df, att_dict):
    ''' Add a feature that is the dew point depression, which is temp - dew. This can be an indicator of how comfortable the weather is for biking.
    '''
    df['dew_point_depression'] = df['temp'] - df['dew']
    att_dict['dew_point_depression'] = 'numerical'    
    return df, att_dict

def one_hot_encode_categorical(df, att_dict):
    ''' One hot encode all categorical features except the target.
    '''
    cat_cols = [col for col in df.columns if att_dict[col] == 'categorical']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    for col in cat_cols:
        del att_dict[col]
    for col in df.columns:
        if col not in att_dict:
            att_dict[col] = 'binary'
    return df, att_dict

def get_ready_extended_data(scaler=StandardScaler()):
    '''För tillfället så fungerar den här funktionen som en samling av all feature engineering'''
    extended_data = get_unscaled_data()
    attribute_type = create_attribute_type_dict()
    ''' How to encode categorials:'''
    extended_data, attribute_type = one_hot_encode_categorical(extended_data, attribute_type)
    
    ''' Add features if wanted'''
    #extended_data, attribute_type = add_app_temp_features(extended_data, attribute_type )
    #extended_data, attribute_type = add_good_weather_feature(extended_data, attribute_type)
    #extended_data, attribute_type = add_is_raining_feature(extended_data, attribute_type)
    #extended_data, attribute_type = add_rush_hour_feature(extended_data, attribute_type)
    #extended_data, attribute_type = add_dew_point_depression_feature(extended_data, attribute_type)
    ''' Drop features if wanted '''
    drop_cols = []
    #drop_cols = ["dew"] # Comment out this line for no dropped features.
    extended_data.drop(columns=drop_cols, inplace=True)
    for col in drop_cols:
        del attribute_type[col]
    
    ''' Scale numerical features '''
    num_cols = [col for col in extended_data.columns if attribute_type[col] == 'numerical']
    for col in num_cols:
        extended_data[col] = scaler.fit_transform(extended_data[[col]])
    return extended_data

if __name__ == "__main__":
    extended_data, attribute_type = add_app_temp_features(extended_data, attribute_type )
    extended_data, attribute_type = add_good_weather_feature(extended_data, attribute_type)
    extended_data, attribute_type = add_is_raining_feature(extended_data, attribute_type)
    extended_data, attribute_type = one_hot_encode_categorical(extended_data, attribute_type)
    ''' # These get kinda messy, the ones below are more readable.
    cat_cols = [col for col in extended_data.columns if attribute_type[col] == 'categorical' or attribute_type[col] == 'binary']
    num_cols = [col for col in extended_data.columns if attribute_type[col] == 'numerical']
    corr_mat(extended_data, num_cols)
    chi2_stats(extended_data, cat_cols)
    '''

    hour_cols = [col for col in extended_data.columns if "hour_of_day" in col]
    day_cols = [col for col in extended_data.columns if "day_of_week" in col]
    month_cols = [col for col in extended_data.columns if "month" in col]
    chi2_stats(extended_data, hour_cols)
    #chi2_stats(extended_data, day_cols)
    #chi2_stats(extended_data, month_cols)