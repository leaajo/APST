import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Charger les données
df = pd.read_csv("second_study/data/AuteursNat.csv")
auteurs = pd.read_csv("second_study/data/Auteurs.csv", sep=" ")

# Sélectionner uniquement les colonnes où la première ligne est "NOM" ou "ADJ"
auteurs_adj = auteurs.loc[:, (auteurs.iloc[0] == "NOM") | (auteurs.iloc[0] == "ADJ")]

# Ajouter la colonne des auteurs
auteurs_adj.insert(0, "voc", auteurs["voc"])
# Supprimer la ligne d'étiquettes grammaticales
auteurs_adj = auteurs_adj.drop(index=0).reset_index(drop=True)

# Définir les caractéristiques et la variable cible
X = auteurs_adj.drop(columns=["voc"]).astype(float)
y = df["Naissance"].astype(float)

# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définition du modèle Ridge
ridge = Ridge()

# Recherche du meilleur hyperparamètre alpha avec validation croisée
param_grid_ridge = {"alpha": [0.006]}
grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_ridge.fit(X_train_scaled, y_train)

# Meilleur modèle et évaluation
ridge_best = grid_search_ridge.best_estimator_
y_pred_ridge = ridge_best.predict(X_test_scaled)

# Calcul des métriques
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = ridge_best.score(X_test_scaled, y_test)

# Validation croisée sur l'ensemble d'entraînement
rmse_ridge_cv = np.sqrt(-cross_val_score(ridge_best, X_train_scaled, y_train, cv=10, scoring='neg_mean_squared_error').mean())
r2_ridge_cv = cross_val_score(ridge_best, X_train_scaled, y_train, cv=10, scoring='r2').mean()

# Affichage des résultats
print(f"Meilleur alpha Ridge : {grid_search_ridge.best_params_['alpha']}")
print(f"RMSE Ridge : {rmse_ridge}")
print(f"R2 Ridge (Test) : {r2_ridge}")
print(f"RMSE Ridge (CV) : {rmse_ridge_cv}")
print(f"R2 Ridge (CV) : {r2_ridge_cv}")