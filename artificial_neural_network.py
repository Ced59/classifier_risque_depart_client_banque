# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import theano as th
import matplotlib.pyplot as plt

tf.__version__

# Partie 1 : Préparation des données

# Importer le dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Création de la matrice des données utiles pour la prédiction
x = dataset.iloc[:, 3:13].values
# Création de la matrice des données de sortie pour l'entrainement
y = dataset.iloc[:, 13].values

# Encodage des données catégoriques (pays et genre)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encodeur du genre
labelencoder_X_gender = LabelEncoder()
x[:, 2] = labelencoder_X_gender.fit_transform(x[:, 2])

# Encodeur du pays
labelencoder_X_Country = LabelEncoder()
x[:, 1] = labelencoder_X_Country.fit_transform(x[:, 1])

# Transformation en variable ordinales du pays
# (les 3 premières colonnes seront 0 ou 1 pour représenter les 3 pays possibles)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# On supprime la première colonne (si 0,0 ce sera ce pays là)
x = x[:, 1:]

# Séparer en un jeu d'entrainement et un jeu de test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Normalisation des variables (tout mettre en valeur entre -2 et 2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Partie 2 : Contruction du réseau de neurones

# Importation des modules de keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialisation
classifier = Sequential()

# Ajout couche d'entrée et couche cachée avec fonction redresseur
classifier.add(
    Dense(units=6,
          activation="relu",
          kernel_initializer="uniform",
          input_dim=11)
)

# Ajout deuxième couche cachée avec fonction redresseur
classifier.add(
    Dense(units=6,
          activation="relu",
          kernel_initializer="uniform")
)

# Ajout couche de sortie avec fonction sigmoïde
classifier.add(
    Dense(units=1,
          activation="sigmoid",
          kernel_initializer="uniform")
)

# Compilation du réseau de neurones
classifier.compile(
    optimizer="adam",  # algorithme gradient stochastique
    loss="binary_crossentropy",  # fonction de coût
    metrics=["accuracy"]
)



