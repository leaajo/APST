import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import csv

# Lecture du fichier CSV compressé directement dans un DataFrame
data = pd.read_csv('data/data.csv.gz', compression='gzip')

# On réduit le df 
data_reduced = data[["Unnamed: 0"]].join(data.sample(n=10, axis=1))
labels = pd.read_csv('data/labels.csv')

df_merged = pd.merge(data_reduced, labels, on="Unnamed: 0", how="inner")  
df_merged.set_index('Unnamed: 0', inplace=True)

sample_10_1 = df_merged.to_csv("data/sample_10_3.csv", sep=",")
