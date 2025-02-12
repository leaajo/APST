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
data_reduced = data.sample(n=10, axis=1)
labels = pd.read_csv('data/labels.csv')
data_reduced['labels']= labels['Class']

sample_10_1 = data_reduced.to_csv("data\sample_10_1", sep=",")
#print(data_reduced.shape)  # Doit afficher (801, 30)

# Heatmap
#plt.figure(figsize=(12, 8))
#sns.heatmap(data_reduced, cmap='viridis')
#plt.title('Heatmap des expressions géniques')
#plt.show()



#fulldf.head()