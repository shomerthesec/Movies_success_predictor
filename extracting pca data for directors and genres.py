# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:23:57 2021

@author: Shomer
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:43:52 2021

@author: Shomer
"""

# %% importings

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
provided_data = pd.read_csv('data/provided_data.csv', index_col=0)

# %% scaling the time of the movies
time_scaler = MinMaxScaler()
times = time_scaler.fit_transform(
    provided_data.loc[:, 'Runtime (Minutes)'].values.reshape(-1, 1)).squeeze()

# %% using PCA on directors names
directors_pca = PCA(n_components=3)

directors = directors_pca.fit_transform(
    pd.get_dummies(provided_data.Director)).squeeze()

# %% using pca om genres
genres_pca = PCA(n_components=3)
genres = genres_pca.fit_transform(provided_data.iloc[:, 3:]).squeeze()

# %% concatinating our data

pca_df = pd.concat([pd.Series(times, name='times'),
                   pd.DataFrame(directors, columns=['directors_c1',
                                                    'directors_c2',
                                                    'directors_c3']),

                   pd.DataFrame(genres, columns=['genres_c1',
                                                 'genres_c2',
                                                 'genres_c3']),
                   provided_data.Success],

                   axis=1
                   )
pca_df.to_csv('data/pca_data.csv')
