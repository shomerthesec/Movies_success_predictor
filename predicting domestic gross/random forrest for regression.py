# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:30:29 2021

@author: Shomer
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# %%
provided_data = pd.read_csv(
    'S:/Movie_success_predictor/data/train_budget_numerical.csv', index_col=0)


# %% let's try to use knn
df_train, df_test = train_test_split(provided_data,
                                     test_size=0.20,
                                     random_state=0,
                                     )
# %%
model = RandomForestRegressor(100)
model.fit(X=df_train.drop('DomesticGross', axis=1),
          y=df_train.DomesticGross)


# %%
predictions = model.predict(df_test.drop('DomesticGross', axis=1))

# %%

mean_squared_error(df_test.DomesticGross,
                   predictions)
# mse = 0.5331216169616148

# %%

r2_score(df_test.DomesticGross,
         predictions)
# r2 = 0.5327573677425546

 # %%
# # let's measure the error given when reversing the power transform -log trans-
# from scipy.special import inv_boxcox
# l= 0.28987197

# predictions_descaled= inv_boxcox( predictions , l)

# actual_descaled= inv_boxcox (df_test.DomesticGross.values , l)

# %% let's try another method for inverse transforming


def return_pwr_trans():
     from sklearn.preprocessing import PowerTransformer
    df = pd.read_csv('data/provided_data_with_budget.csv',
                 index_col=0)

    df.drop(index=df.query('DomesticGross==0').index,
        inplace=True)
    boxcox_gross = PowerTransformer(method='box-cox')
    train_df, _ = train_test_split(df, test_size=0.2, random_state=0)
    boxcox_gross.fit_transform(
            train_df.loc[:, 'DomesticGross'].values.reshape(-1, 1) )
    return boxcox_gross

# %%

transform=return_pwr_trans()

actual_descaled= transform.inverse_transform( df_test.DomesticGross.values.reshape(-1,1) )
predictions_descaled= transform.inverse_transform(predictions.reshape(-1,1))
# %%
mean_squared_error(actual_descaled , predictions_descaled )

# %% plotting the real values vs the predicted valeus, in a scale of 100 millions
plt.plot(actual_descaled,
             label='Actual')

plt.plot(predictions_descaled,
             label='Predicted')
plt.legend()

# %% the performance is not bad at ALL! we can see the model is even less over shooting
# let's try to see the performance on unseen data
test_data= pd.read_csv('S:/Movie_success_predictor/data/test_budget_numerical.csv', index_col=0)

test_predictions = model.predict(test_data.drop('DomesticGross', axis=1))

# %%

mean_squared_error(test_data.DomesticGross,
                   test_predictions)
# mse = 0.4681593181867785

# %%

r2_score(test_data.DomesticGross,
         test_predictions)
# r2 = 0.4411042729283037

# %% let's visualize the predictions after descaling

test_actual_descaled= transform.inverse_transform( test_data.DomesticGross.values.reshape(-1,1) )

test_predictions_descaled= transform.inverse_transform(test_predictions.reshape(-1,1))

# %%

plt.plot(test_actual_descaled,
             label='test_Actual')

plt.plot(test_predictions_descaled,
             label='test_Predicted')
plt.legend()

# %% let's try distplot plot

sns.distplot(
             test_actual_descaled,
             label='test_Actual')

sns.distplot(
             test_predictions_descaled,
             label='test_Predicted')

plt.legend()

# %% let's try hist plots

plt.hist(
             test_actual_descaled,
             bins=136,
             label='test_Actual')

plt.hist(
             test_predictions_descaled,
             bins=136,
             label='test_Predicted')

plt.legend()

#%% let's plot the transformed data


sns.distplot(
             test_data.DomesticGross,
             label='test_Actual_scaled')

sns.distplot(
             test_predictions,
             label='test_Predicted_scaled')

plt.legend()