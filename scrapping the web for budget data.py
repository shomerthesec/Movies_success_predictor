# %% importings
import pandas as pd
import numpy as np
import requests

# # %% requesting the data from the url
# resp = requests.get(url='https://www.the-numbers.com/movie/budgets/all')


# # %% checking the headers of the url
# for i in resp.headers:
#     print(resp.headers.get(i))
# # %% getting the data from resp.text as html and reading it using pandas
# data = pd.read_html(resp.text)
# # %% ok now we need to read punch of urls to have more data and then stacking them
# urls = ['https://www.the-numbers.com/movie/budgets/all/101',
#         'https://www.the-numbers.com/movie/budgets/all/201',
#         'https://www.the-numbers.com/movie/budgets/all/301',
#         'https://www.the-numbers.com/movie/budgets/all/401',
#         'https://www.the-numbers.com/movie/budgets/all/501',
#         'https://www.the-numbers.com/movie/budgets/all/601',
#         'https://www.the-numbers.com/movie/budgets/all/701',
#         ]
# # %% web scrapping, to create dataframe of the budget for all movies
# data = pd.DataFrame([])

# for url in urls:
#     r = requests.get(url)
#     data = pd.concat([data, pd.read_html(r.text)[0]], axis=0)

# # %%
# data['Title'] = data.Movie

# %% reading our existing data
df = pd.read_csv('data/movie_success_rate.csv', index_col=0)
# # %%
# joined = df.merge(data, on='Title')

# %% trying a for loop to get more pages

# url = 'https://www.the-numbers.com/movie/budgets/all/'
# for i in range(63):
#     print(url + str(i*10) + '1')
#     # SMART A$$

# %% we are going to use that to scrape all the movies
data = pd.DataFrame([])
url = 'https://www.the-numbers.com/movie/budgets/all/'
for i in range(63):
    r = requests.get(url + str(i*10) + '1')
    data = pd.concat([data, pd.read_html(r.text)[0]], axis=0)

# %% let's join our data
joined = df.merge(data, left_on='Title', right_on='Movie',
                  ).drop_duplicates(['Title']).drop(['Unnamed: 0',
                                                    'ReleaseDate',
                                                     'Movie',
                                                     'Genre'], axis=1)


# %% to get the data we need we will drop the unnessary columns
list_columns = ['Year',
                'Rating',
                'Votes',
                'Revenue (Millions)',
                'Metascore',
                'WorldwideGross',
                ]
final_data = joined.drop(columns=list_columns)

# %%
# dealing with the dollar signs in the budget columns


def convert_dollars(df,
                    col_list=['ProductionBudget', 'DomesticGross']):

    for col in col_list:
        df[col] = df[col].str.replace(',', '')
        df[col] = df[col].str.replace('$', '')
        df[col] = df[col].astype(float)
    return


# %%
convert_dollars(joined, ['ProductionBudget',
                'DomesticGross', 'WorldwideGross'])
convert_dollars(final_data)


# %% saving the data
final_data.to_csv('data/provided_data_with_budget.csv')

joined.to_csv('data/movie_success_rate_with_budgets.csv')
