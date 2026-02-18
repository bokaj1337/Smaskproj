import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
from load_data import get_ready_data, create_attribute_type_dict

def corr_mat(df, att_dict):
    # Check correlation matrix of numerical attributes and target
    num_cols = [col for col in df.columns if att_dict[col] == 'numerical']
    corr =  df[num_cols+["increase_stock"]].corr()
    plt.figure(figsize=(14,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def chi2_stats(df, att_dict):
    # Check chi2 of categorical attributes
    cat_cols = [col for col in df.columns if att_dict[col] == 'categorical']
    X = df[cat_cols]
    y = df["increase_stock"]
    chi_scores = chi2(X, y)

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
    ax2.bar_label(bar2, padding=3, fmt='%.2f')

    ax1.set_xticks(x)
    ax1.set_xticklabels(X.columns)
    bars = [bar1,bar2]
    labels_legend = [b.get_label() for b in bars]
    ax1.legend(bars, labels_legend, loc='upper right')
    plt.show()


if __name__ == "__main__":
    # Get the standardscaled data
    std_data = get_ready_data()
    attribute_type = create_attribute_type_dict(std_data)
    
    corr_mat(std_data, attribute_type)
    chi2_stats(std_data, attribute_type)
    ''' Observations:
    - Corr matrix shows that temp and dew have high corr with target. However, they
    are also highly correlated with each other, so we might want to drop one of them.

    
    '''