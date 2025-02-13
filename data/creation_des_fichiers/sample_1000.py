import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import csv

# Lecture du fichier CSV compressé directement dans un DataFrame
data = pd.read_csv('data/data.csv.gz', compression='gzip')
labels = pd.read_csv('data/labels.csv')

for i in range(1,6):
# On réduit le df 
    data_reduced = data[["Unnamed: 0"]].join(data.sample(n=1000, axis=1))
    df_merged = pd.merge(data_reduced, labels, on="Unnamed: 0", how="inner")  
    df_merged.set_index('Unnamed: 0', inplace=True)
    
    output_path = f"data/sample_1000_{i}.csv"  # Création du nom dynamique
    df_merged.to_csv(output_path, sep=",")