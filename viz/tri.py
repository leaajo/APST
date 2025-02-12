import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import csv

# Lecture du fichier CSV compressé directement dans un DataFrame
data = pd.read_csv('data/data.csv.gz', compression='gzip')
data.set_index('Unnamed: 0', inplace=True)

# On réduit le df 
data_reduced = data.sample(n=30, axis=1)

print(data_reduced.shape)  # Doit afficher (801, 30)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_reduced, cmap='viridis')
plt.title('Heatmap des expressions géniques')
plt.show()

labels = pd.read_csv('data/labels.csv')
fulldf = data_reduced 
fulldf['labels']= labels['Class']

fulldf.head()