# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv(
    'S:/Movie_success_predictor/data/movie_success_rate.csv', skipfooter=1)

# %%
describtion = data.describe()


# %%

for i, col in enumerate(describtion.columns[:7]):
    _, ax = plt.subplots(1, 1)
    sns.distplot(data[col], ax=ax)


# %%
data['Revenue (Millions)']  # skewed to the right

# %% mimicing the distribution of Metascore with a normal dist
mu, sig = data.Metascore.mean(), data.Metascore.std()
sns.distplot(np.random.normal(mu, sig, 1000))

# %% dropping genre as it's already converted into dummy columns
df = data.copy()  # .drop('Genre', axis=1)

# %% appending the names of unique actors to a list

all_actors = []
for n in df.Actors:
    for n_i in n.split(','):
        all_actors.append(n_i.strip())

all_actors = pd.unique(all_actors)

# %%

# tmp_df=pd.DataFrame([], columns=all_actors)

# for actor_name in all_actors:
#    for idx, n_i in enumerate(df.Actors):
#        tmp_df[actor_name] = actor_name in n_i


# %%
categories = data.columns[12:-1]

# %%
for cat in categories:
    print(data[]

# %% number of each category
data[categories].sum(axis=0).plot.bar()

# %% action movies that succeeded

sns.countplot(x=data.query('Action==1').Success,
              )
plt.title('Action films and their success')

# %%
sns.countplot(x=data.query('Aniimation==1').Success)
plt.title('Animation films and their success')

# %%
sns.countplot(x=data.query('Drama==1').Success)
plt.title('Drama films and their success')

# %%
zero_matrix=np.zeros((len(df), len(all_actors)))

actors_df=pd.DataFrame(zero_matrix, columns=all_actors)

for i, actor_names in enumerate(df.Actors):

    tmp_names=[]

    for name in actor_names.split(','):
         tmp_names.append(name.strip())

    indices=actors_df.columns.get_indexer(tmp_names)
    actors_df.iloc[i, indices]=1

# %%
# dummies is a dataframe for actors names

# %%
sns.catplot(x='Success',
            col='Genre',
            data=data,
            hue="Genre",
            kind='count')

# %%
sns.relplot(data=df,
            x="Success",
            y=categories.T)

# %%
# data.groupby('Category')
# can't do a group by category as it's splitted into multi genres

# %% to return the data in a certain category
# for category in categories:
#    # mask= df.loc[:,category] == 1
#     tmp= df[ df.loc[:,category]  ==1 ]

# %% function to do the ubove task
def category_df(df, category):
    return df.query(f'{category}==1')

# %% to only get the action moveis
# action_df= category_df(df , 'Action' )

# %% to join names of the actors with the movies
# display( action_df.join(actors_df, on= action_df.index ) )

# %% duo to the highly sparse actors list let's work on the directors

provided_data=df.drop(['Actors', 'Title', 'Rank', 'Year', 'Rating',
                        'Votes', 'Revenue (Millions)', 'Metascore'],
                        axis=1)

# %% let's group by director and see what we can find

dir_tmp=provided_data.groupby('Director').sum()
 # we can see that we have 524 unique director, with some of them so successful

# %% shows the directos and their success
succ_tmp=dir_tmp.query('Success != 0')
sns.barplot(x=succ_tmp.index, y=succ_tmp.Success)

# %% saving our processed data
provided_data.to_csv('data/provided_data.csv')
actors_df.to_csv('data/actors_data.csv')
