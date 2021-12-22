# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:44:19 2021

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
from sklearn.ensemble import RandomForestClassifier

# %% loading data
provided_data = pd.read_csv('data/pca_data.csv', index_col=0)

# %%

rf_train, rf_test = train_test_split(provided_data,
                                     test_size=0.20,
                                     random_state=0,
                                     )
# %%
rf_clf = RandomForestClassifier(100, random_state=0)
rf_clf.fit(rf_train.drop('Success', axis=1),
           rf_train.Success)
# %%
rf_pred = rf_clf.predict(rf_test.drop('Success', axis=1))

# %%
f1_score(rf_test.Success, rf_pred)
# a f1 of 0.76 is not bad at all

# %%

nbconv_matrix = confusion_matrix(rf_test.Success, rf_pred)

conv_matrix = ConfusionMatrixDisplay(nbconv_matrix)

conv_matrix.plot()

# %%
nbconv_matrix = confusion_matrix(rf_train.Success,
                                 rf_clf.predict(rf_train.drop('Success',
                                                              axis=1)))

conv_matrix = ConfusionMatrixDisplay(nbconv_matrix)

conv_matrix.plot()

# we can see clearly that the model over fitted the training data,
# but the predictions are not bad at all!
