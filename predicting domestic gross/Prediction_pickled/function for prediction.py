import pickle
import numpy as np
import pandas as pd

model= pickle.load(open('model.sav', 'rb'))
production_scale = pickle.load(open('production_scale.sav', 'rb'))
gross_scale = pickle.load(open('gross_scale.sav', 'rb'))
runtime_scale = pickle.load(open('runtime_scale.sav', 'rb'))
genres_pca = pickle.load(open('genres_pca.sav', 'rb'))

def full_genres( given  ):
    genres = np.array( ['action', 'adventure', 'aniimation', 'biography', 'comedy',
                        'crime', 'drama', 'family', 'fantasy', 'history', 'horror',
                        'music', 'musical', 'mystery', 'romance', 'sci-fi', 'sport',
                        'thriller', 'war', 'western'] )
    given = [cat.lower() for cat in given]
    genre_ip= np.zeros_like(genres, dtype=int)
    idx= [ np.where(genres == g )[0].tolist()[0] for g in given ] # returns the idx for the given genres
    genre_ip[idx]=1
    return genre_ip

def predict_gross( data ):
    data['runtime']= runtime_scale.transform( np.array( data['runtime'] ).reshape(1, -1) )[0]
    data['production']= production_scale.transform( np.array( data['production']).reshape(1, -1) )[0]
    data['c1'], data['c2'], data['c3'] = zip( *genres_pca.transform( full_genres( data['genre'] ).reshape(1, -1).tolist() ) )
    del data['genre']
    return gross_scale.inverse_transform( model.predict( pd.DataFrame(data)).reshape(1, -1)  ).tolist()[0][0]


data={}
data['runtime']= input('Enter runtime\n')
data['production']= input('Enter production price\n')
print('\nEnter genres carefully from here: \naction, adventure, aniimation, biography,\ncomedy, crime, drama, family, \nfantasy, history, horror, musis, \nmusical, mystery, romance, sci-fi, \nsport,thriller, war, western\n' )
data['genre']= [input('Enter genre\n') for i in range(3)]

print( f'\nPredicted gross= {np.floor(predict_gross(data))} $')
#%%
