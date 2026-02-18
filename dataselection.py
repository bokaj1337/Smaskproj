import seaborn as sns
import matplotlib.pyplot as plt
from load_data import get_ready_data

cleaned_fil = get_ready_data()

plt.figure(figsize=(12, 8))                 # Bygger korrelations matriser för att se vilka som spelar mest roll på resultatet. 
correlation = cleaned_fil.corr()
sns.heatmap(correlation[['increase_stock']].sort_values(by='increase_stock', ascending=False), annot=True, cmap='RdYlGn')
plt.show()

