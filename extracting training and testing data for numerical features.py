# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 14:01:21 2021

@author: Shomer
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

# %%
df = pd.read_csv('data/provided_data_with_budget.csv',
                 index_col=0)

df.drop(index=df.query('DomesticGross==0').index,
        inplace=True)
df.drop('Success', axis=1, inplace=True)
# %% loading a power transformer to transform the budgets 'log transform'
boxcox_production = PowerTransformer(method='box-cox')
boxcox_gross = PowerTransformer(method='box-cox')
# %% loading a minmax scaler for time of the movie
minmax = MinMaxScaler()

# %% loading a PCA to reduce the genre col
pca = PCA(n_components=3)

# %% the genre columns
genre_cols = df.iloc[:, 5:25].columns

# %% let's devide our data now

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# %% lets transform our production budget

train_df.loc[:, 'ProductionBudget'] = boxcox_production.fit_transform(
    train_df.loc[:, 'ProductionBudget'].values.reshape(-1, 1)
)
# %% lets transform our gross budget

train_df.loc[:, 'DomesticGross'] = boxcox_gross.fit_transform(
    train_df.loc[:, 'DomesticGross'].values.reshape(-1, 1)
)

# %% let's transform the same col for the test_df

test_df.loc[:, 'ProductionBudget'] = boxcox_production.transform(
    test_df.loc[:, 'ProductionBudget'].values.reshape(-1, 1)
)
test_df.loc[:, 'DomesticGross'] = boxcox_gross.transform(
    test_df.loc[:, 'DomesticGross'].values.reshape(-1, 1)
)

# %% let's transform the time of the movie

train_df.loc[:, 'Runtime (Minutes)'] = minmax.fit_transform(
    train_df.loc[:, 'Runtime (Minutes)'].values.reshape(-1, 1)
)
test_df.loc[:, 'Runtime (Minutes)'] = minmax.transform(
    test_df.loc[:, 'Runtime (Minutes)'].values.reshape(-1, 1)
)

# %% use pca transformation on the genre cols [5:25]

train_pca = pca.fit_transform(
    train_df.iloc[:, 5:25]
)
test_pca = pca.transform(
    test_df.iloc[:, 5:25]
)

# %% let's join the train_df with the pca
train_data = train_df.drop(genre_cols, axis=1).join(
    pd.DataFrame(train_pca, index=train_df.index,
                 columns=['c1', 'c2', 'c2'])
)
test_data = test_df.drop(genre_cols, axis=1).join(
    pd.DataFrame(test_pca, index=test_df.index,
                 columns=['c1', 'c2', 'c2'])
)
# %%
test_data.iloc[:, 4:].to_csv('data/test_budget_numerical.csv')
train_data.iloc[:, 4:].to_csv('data/train_budget_numerical.csv')

# %%
boxcox_gross.lambdas_
