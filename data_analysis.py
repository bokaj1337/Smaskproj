import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
from load_data import get_ready_data


# Get the standardscaled data
std_data = get_ready_data()
num_cols = ["temp","dew","humidity","precip","snowdepth","windspeed","cloudcover","visibility"]
cat_cols = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']

def corr_mat():
    # Check correlation matrix of numerical attributes and target
    corr =  std_data[num_cols+["increase_stock"]].corr()
    plt.figure(figsize=(14,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def chi2_stats():
    # Check chi2 of categorical attributes
    X = std_data[cat_cols]
    y = std_data["increase_stock"]
    chi_scores = chi2(X, y)

    print(chi_scores)

    x = np.arange(len(X.columns))
    fig, ax1 = plt.subplots(figsize=(14,10))
    width=0.35
    ax1.set_xlabel("Attribute")
    ax1.set_ylabel("Chi Score")
    bar1 = ax1.bar(x-width/2, chi_scores[0], width, label="Chi Score", color="blue")
    ax1.bar_label(bar1, padding=3, fmt='%.2f')

    ax2 = ax1.twinx()
    bar2 = ax2.bar(x+width/2, chi_scores[1], width, label="P Score", color="orange")
    ax2.set_ylabel("P Score")
    ax2.bar_label(bar1, padding=3, fmt='%.2f')

    ax1.set_xticks(x)
    ax1.set_xticklabels(X.columns)
    bars = [bar1,bar2]
    labels_legend = [b.get_label() for b in bars]
    ax1.legend(bars, labels_legend, loc='upper left')
    
    plt.show()

    



if __name__ == "__main__":
    #corr_mat()
    chi2_stats()