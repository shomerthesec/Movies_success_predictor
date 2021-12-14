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
provided_data = pd.read_csv('data/provided_data.csv', index_col=0)

# %%
df = pd.get_dummies(provided_data.Director)
svm_linear_df = provided_data.drop(
    ['Description', 'Director'], axis=1).join(df)


# %% let's try to use knn
l_svm_train, l_svm_test = train_test_split(svm_linear_df,
                                           test_size=0.20,
                                           random_state=0,
                                           )
# %%
l_svm = SVC(C=0.9, kernel='linear')
l_svm.fit(X=l_svm_train.drop('Success', axis=1),
          y=l_svm_train.Success)


# %%
l_svm_pred = l_svm.predict(l_svm_test.drop('Success', axis=1))

# # %%
# l_svm.score(l_svm_test.drop('Success', axis=1),
#           l_svm_test.Success)


# %%
f1_score(l_svm_pred, l_svm_test.Success)
# a f1 of 0.33 is really bad

# %%

l_svm_conv_matrix = confusion_matrix(l_svm_pred, l_svm_test.Success)

conv_matrix = ConfusionMatrixDisplay(l_svm_conv_matrix)

conv_matrix.plot()

# we can see that using the knn without the directors is really bad
