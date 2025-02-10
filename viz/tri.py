import numpy as np
import pandas as pd
import matplotlib as plt
import gzip
import csv

# Décompression du fichier data :
with gzip.open('data/data.csv.gz', 'rt', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
