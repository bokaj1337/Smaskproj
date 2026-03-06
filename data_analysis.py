import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
from load_data import get_ready_data, create_attribute_type_dict

def corr_mat(df, features):
    # Check correlation matrix of numerical attributes and target
    corr =  np.round(df[features+["increase_stock"]].corr(), 2)
    plt.figure(figsize=(6,5))
    plt.title("Correlation Matrix")
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.xticks(rotation=25, ha='right')
    plt.savefig("figures/correlation_matrix.png", bbox_inches='tight',)
    plt.show()

def chi2_stats(df, features):
    # Check chi2 of categorical attributes
    X = df[features]
    y = df["increase_stock"]
    chi_scores = chi2(X, y)

    x = np.arange(len(X.columns))
    fig, ax1 = plt.subplots(figsize=(6,5))
    width=0.3
    ax1.set_xlabel("Attribute")
    ax1.set_ylabel("Chi Score")
    bar1 = ax1.bar(x-width/2, chi_scores[0], width, label="Chi Score", color="blue")
    ax1.bar_label(bar1, padding=3, fmt='%.2f')

    ax2 = ax1.twinx()
    bar2 = ax2.bar(x+width/2, chi_scores[1], width, label="P Score", color="orange")
    ax2.set_ylabel("P Score")
    ax2.bar_label(bar2, padding=3, fmt='%.2f')

    ax1.set_xticks(x)
    ax1.set_xticklabels(X.columns) #ax1.set_xticklabels(X.columns, rotation=20, ha='right')
    bars = [bar1,bar2]
    labels_legend = [b.get_label() for b in bars]
    ax1.legend(bars, labels_legend, loc='upper center')

    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.10) 
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.10)

    plt.title("Chi2 and P Scores for Categorical Features")
    plt.tight_layout()
    plt.savefig("figures/chi2_scores.png", bbox_inches='tight',)
    plt.show()


def plot_temporal_demand(dataframe):
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    hourly = dataframe.groupby("hour_of_day")["increase_stock"].mean().reset_index()
    ax1.plot(hourly["hour_of_day"], hourly["increase_stock"])
    ax1.grid()
    ax1.set_title("Demand over hours of day")

    weekday = dataframe.groupby("day_of_week")["increase_stock"].mean().reset_index()  
    ax2.plot(weekday["day_of_week"], weekday["increase_stock"])  
    ax2.set_title("Demand over day of week") 
    ax2.grid()

    monthly = dataframe.groupby("month")["increase_stock"].mean().reset_index()  
    ax3.plot(monthly["month"], monthly["increase_stock"]) 
    ax3.set_title("Demand over month")
    ax3.grid()
    plt.tight_layout()
    plt.savefig("figures/temporal_demand.png", bbox_inches='tight',)
    plt.show()
    plt.figure(figsize=(6,5))


if __name__ == "__main__":
    # Get the standardscaled data
    std_data = get_ready_data()
    attribute_type = create_attribute_type_dict()

    num_cols = [col for col in std_data.columns if attribute_type[col] == 'numerical']
    cat_cols = [col for col in std_data.columns if attribute_type[col] == 'categorical' or attribute_type[col] == 'binary']

    #corr_mat(std_data, num_cols)
    #corr_mat(std_data, cat_cols)
    #chi2_stats(std_data, ['holiday', 'weekday', 'summertime'])
    plot_temporal_demand(std_data)
    ''' Observations:
    - Corr matrix shows that temp and dew have high corr with target. However, they
    are also highly correlated with each other, so we might want to drop one of them.
    '''