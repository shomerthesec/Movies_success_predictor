# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
data = pd.read_csv(
    'S:/Movie_success_predictor/data/movie_success_rate_with_budgets.csv', index_col=0)
# %%
cols = ['ProductionBudget', 'DomesticGross',
        'WorldwideGross']
for i, col in enumerate(cols):
    _, ax = plt.subplots(1, 1)
    sns.distplot((data[col]), ax=ax)


# %%
for i, col in enumerate(cols):
    _, ax = plt.subplots(1, 1)
    sns.distplot(np.log(data[col]+1e-9), ax=ax)
# adding 1*10^-19 just to make sure that log(0) doesn't return an error
# %%
# we can further look for removing the outliers that result in 0
idx_to_drop = data.query('DomesticGross==0').index
data.drop(index=idx_to_drop, inplace=True)

# %%
# let's plot again
for col in cols:
    _, ax = plt.subplots(1, 1)
    sns.distplot(np.log(data[col]), ax=ax)
# it seems better, with some skewing to the left, but let's keep
# it like that to avoid over filtration
# %%
sns.regplot('ProductionBudget', 'DomesticGross', data=data)

plt.title('Production Budget vs Domestic Gross')
# from here we can clearly see the pattern when the budget gets
# higher the gross is higher

# %%

sns.scatterplot('ProductionBudget', 'DomesticGross',
                data=data,
                hue='Director',
                )
plt.legend([], [], frameon=False)
# to remove the legend as it's huge
# %%

sns.regplot('ProductionBudget', 'DomesticGross',
            data=data.query('Action==1'))

plt.title('Action movies')

# %%
genres = data.iloc[:, 10:31].columns

# %%
for genre in genres:

    _, ax = plt.subplots(1, 1)

    sns.regplot('ProductionBudget', 'DomesticGross',
                data=data.query(f'{genre}==1'),
                # it works but not for sci-fi
                ax=ax)

    plt.title(f'{genre} movies')

# %%
for genre in genres:

    _, ax = plt.subplots(1, 1)

    sns.regplot('ProductionBudget', 'DomesticGross',
                data=data[data.loc[:, genre] == 1],
                # it works but not for sci-fi
                ax=ax)

    plt.title(f'{genre} movies')
