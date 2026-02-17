import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

raw = pd.read_csv("training_data_VT2026.csv")

def remove_constant_data(df):
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    print(f"Dropping constant columns: {constant_columns}")
    return df.drop(constant_columns, axis=1)    

def process_data(df, scaler=StandardScaler()):
    '''
    Makes stock_demand binary and scales data.
    '''
    out = df.replace({"low_bike_demand": 0, "high_bike_demand": 1})
    numerical_columns = ["temp","dew","humidity","precip","snowdepth","windspeed","cloudcover","visibility"]
    for col in numerical_columns:
        out[col] = scaler.fit_transform(out[[col]])
    return out


fil = remove_constant_data(raw)
cleaned_fil = process_data(fil)


def plot_hourly_avg(dataframe):
    hourly_avg = dataframe.groupby("hour_of_day")[["temp","dew","humidity","precip","snowdepth","windspeed","cloudcover","visibility","increase_stock"]].mean().reset_index()
    plt.figure(figsize=(12, 8))
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["temp"], label="Temperature")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["dew"], label="Dew Point")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["humidity"], label="Humidity")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["precip"], label="Precipitation")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["snowdepth"], label="Snow Depth")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["windspeed"], label="Wind Speed")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["cloudcover"], label="Cloud Cover")
    plt.plot(hourly_avg["hour_of_day"], hourly_avg["visibility"], label="Visibility")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Value")
    plt.title("Average Weather Conditions by Hour of Day")
    plt.legend()
    plt.grid()
    plt.show()
# plot_hourly_avg(cleaned_fil)

def plot_weekday_avg(dataframe):
    weekday_avg = dataframe.groupby("day_of_week")[["temp","dew","humidity","precip","snowdepth","windspeed","cloudcover","visibility"]].mean().reset_index()
    plt.figure(figsize=(12, 8))
    plt.plot(weekday_avg["day_of_week"], weekday_avg["temp"], label="Temperature")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["dew"], label="Dew Point")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["humidity"], label="Humidity")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["precip"], label="Precipitation")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["snowdepth"], label="Snow Depth")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["windspeed"], label="Wind Speed")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["cloudcover"], label="Cloud Cover")
    plt.plot(weekday_avg["day_of_week"], weekday_avg["visibility"], label="Visibility")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Value")
    plt.title("Average Weather Conditions by Day of Week")
    plt.legend()
    plt.grid()
    plt.show()
# plot_weekday_avg(cleaned_fil)

def plot_monthly_avg(dataframe):
    monthly_avg = dataframe.groupby("month")[["temp","dew","humidity","precip","snowdepth","windspeed","cloudcover","visibility"]].mean().reset_index()
    plt.figure(figsize=(12, 8))
    plt.plot(monthly_avg["month"], monthly_avg["temp"], label="Temperature")
    plt.plot(monthly_avg["month"], monthly_avg["dew"], label="Dew Point")
    plt.plot(monthly_avg["month"], monthly_avg["humidity"], label="Humidity")
    plt.plot(monthly_avg["month"], monthly_avg["precip"], label="Precipitation")
    plt.plot(monthly_avg["month"], monthly_avg["snowdepth"], label="Snow Depth")
    plt.plot(monthly_avg["month"], monthly_avg["windspeed"], label="Wind Speed")
    plt.plot(monthly_avg["month"], monthly_avg["cloudcover"], label="Cloud Cover")
    plt.plot(monthly_avg["month"], monthly_avg["visibility"], label="Visibility")
    plt.xlabel("Month")
    plt.ylabel("Average Value")
    plt.title("Average Weather Conditions by Month")
    plt.legend()
    plt.grid()
    plt.show()
# plot_monthly_avg(cleaned_fil)

def plot_temporal_demand(dataframe):
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle('Vertically stacked subplots')

    hourly = dataframe.groupby("hour_of_day")["increase_stock"].mean().reset_index()
    ax1.plot(hourly["hour_of_day"], hourly["increase_stock"])
    ax1.set_title("Demand over hours of day")

    weekday = dataframe.groupby("day_of_week")["increase_stock"].mean().reset_index()  
    ax2.plot(weekday["day_of_week"], weekday["increase_stock"], label = "Demand over day of week")  
    ax2.set_title("Demand over day of week") 

    monthly = dataframe.groupby("month")["increase_stock"].mean().reset_index()  
    ax3.plot(monthly["month"], monthly["increase_stock"]) 
    ax3.set_title("Demand over month")
    plt.show()    
# plot_temporal_demand(cleaned_fil)








