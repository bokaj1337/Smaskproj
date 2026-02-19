import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
from load_data import get_ready_data, create_attribute_type_dict

def corr_mat(df, features):
    # Check correlation matrix of numerical attributes and target
    corr =  df[features+["increase_stock"]].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def chi2_stats(df, features):
    # Check chi2 of categorical attributes
    X = df[features]
    y = df["increase_stock"]
    chi_scores = chi2(X, y)

    x = np.arange(len(X.columns))
    fig, ax1 = plt.subplots(figsize=(12,8))
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
    ax1.set_xticklabels(X.columns, rotation=45, ha='right')
    bars = [bar1,bar2]
    labels_legend = [b.get_label() for b in bars]
    ax1.legend(bars, labels_legend, loc='upper right')
    
    plt.show()


if __name__ == "__main__":
    # Get the standardscaled data
    std_data = get_ready_data()
    attribute_type = create_attribute_type_dict(std_data)

    num_cols = [col for col in std_data.columns if attribute_type[col] == 'numerical']
    cat_cols = [col for col in std_data.columns if attribute_type[col] == 'categorical' or attribute_type[col] == 'binary']

    corr_mat(std_data, num_cols)
    chi2_stats(std_data, cat_cols)
    ''' Observations:
    - Corr matrix shows that temp and dew have high corr with target. However, they
    are also highly correlated with each other, so we might want to drop one of them.

    
    '''