# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:43:52 2021

@author: Shomer
"""

# %% importings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# %%
provided_data = pd.read_csv('data/provided_data.csv', index_col=0)

# %%
knn_df = provided_data.drop(['Description', 'Director'], axis=1)


# %% let's try to use knn
knn_train, knn_test = train_test_split(knn_df,
                                       test_size=0.20)
# %%
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X=knn_train.drop('Success', axis=1),
        y=knn_train.Success)


# %%
knn_pred = knn.predict(knn_test.drop('Success', axis=1))

# %%
knn.score(knn_test.drop('Success', axis=1),
          knn_test.Success)


# %%
f1_score(knn_pred, knn_test.Success)
# a f1 of 0.41 is really bad

# %%

knn_conv_matrix = confusion_matrix(knn_pred, knn_test.Success)

conv_matrix = ConfusionMatrixDisplay(knn_conv_matrix)

conv_matrix.plot()

# we can see that using the knn without the directors is really bad
