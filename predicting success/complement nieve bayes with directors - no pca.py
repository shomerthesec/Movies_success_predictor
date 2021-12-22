# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:23:57 2021

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
from sklearn.preprocessing import MinMaxScaler
provided_data = pd.read_csv('data/provided_data.csv', index_col=0)

#%%
time_scaler= MinMaxScaler()
provided_data.loc[:, 'Runtime (Minutes)'] = time_scaler.fit_transform(
        provided_data.loc[:, 'Runtime (Minutes)'].values.reshape(-1, 1))

# %%
df = pd.get_dummies(provided_data.Director)
nb_df = provided_data.drop(
    ['Description', 'Director'], axis=1).join(df)

# %% let's try to use knn
nb_train, nb_test = train_test_split(nb_df,
                                           test_size=0.20,
                                           random_state=0,
                                           )
# %%
from sklearn.naive_bayes import ComplementNB
naieve = ComplementNB()
naieve.fit(X=nb_train.drop('Success', axis=1),
          y=nb_train.Success)


# %%
nb_pred = naieve.predict(nb_test.drop('Success', axis=1))

# # %%
# l_svm.score(l_svm_test.drop('Success', axis=1),
#           l_svm_test.Success)


# %%
f1_score(nb_test.Success, nb_pred)
# a f1 of 0.305 is really bad, but at least it's not biased

# %%

nbconv_matrix = confusion_matrix(nb_test.Success, nb_pred )

conv_matrix = ConfusionMatrixDisplay(nbconv_matrix)

conv_matrix.plot()

#%%
nbconv_matrix = confusion_matrix(nb_train.Success, naieve.predict(nb_train.drop('Success',
                                                                                axis=1)))

conv_matrix = ConfusionMatrixDisplay(nbconv_matrix)

conv_matrix.plot()