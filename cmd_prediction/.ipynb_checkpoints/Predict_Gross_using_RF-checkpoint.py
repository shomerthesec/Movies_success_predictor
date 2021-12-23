import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, MinMaxScaler


def full_genres(given):
    genres = np.array(['action', 'adventure', 'aniimation', 'biography', 'comedy',
                       'crime', 'drama', 'family', 'fantasy', 'history', 'horror',
                       'music', 'musical', 'mystery', 'romance', 'sci-fi', 'sport',
                       'thriller', 'war', 'western'])
    genre_ip = np.zeros_like(genres, dtype=int)

    idx = [np.where(genres == g.lower())[0].tolist()[0]
           for g in given]  # returns the idx for the given genres

    genre_ip[idx] = 1

    return genre_ip


def predict_gross(data):

    df = pd.read_csv('provided_data_with_budget.csv',
                     index_col=0)
    df.drop(index=df.query('DomesticGross==0').index,
            inplace=True)
    df.drop('Success', axis=1, inplace=True)
    df, _ = train_test_split(df, test_size=0.2, random_state=0)

    boxcox_production = PowerTransformer(method='box-cox')
    boxcox_gross = PowerTransformer(method='box-cox')
    minmax = MinMaxScaler()
    pca = PCA(n_components=3)

    boxcox_production.fit(df.loc[:, 'ProductionBudget'].values.reshape(-1, 1))
    boxcox_gross.fit(df.loc[:, 'DomesticGross'].values.reshape(-1, 1))
    minmax.fit(df.loc[:, 'Runtime (Minutes)'].values.reshape(-1, 1))
    pca.fit(df.iloc[:, 5:25])

    model = RandomForestRegressor(10)

    df = pd.read_csv(
        'S:/Movie_success_predictor/data/train_budget_numerical.csv', index_col=0)
    model.fit(X=df.drop('DomesticGross', axis=1),  y=df.DomesticGross)

    data['runtime'] = minmax.transform(
        np.array(data['runtime']).reshape(1, -1))[0]
    data['production'] = boxcox_production.transform(
        np.array(data['production']).reshape(1, -1))[0]
    data['c1'], data['c2'], data['c3'] = zip(
        *pca.transform(full_genres(data['genre']).reshape(1, -1).tolist()))

    #zip( *pca.transform( np.array( data['genre']).reshape(1, -1) ))

    del data['genre']

    return boxcox_gross.inverse_transform(model.predict(pd.DataFrame(data)).reshape(1, -1)).tolist()[0][0]


data = {}
data['runtime'] = input('Enter runtime\n')
data['production'] = input('Enter production price\n')
print('\nEnter genres carefully from here: \naction, adventure, aniimation, biography,\ncomedy, crime, drama, family, \nfantasy, history, horror, musis, \nmusical, mystery, romance, sci-fi, \nsport,thriller, war, western\n')
data['genre'] = [input('Enter genre\n') for i in range(3)]

print(f'\nPredicted gross= {np.floor(predict_gross(data))} $')
