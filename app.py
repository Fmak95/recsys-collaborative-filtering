from jikanpy import Jikan
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import time
import pdb

def prep_data(username):

    print("Loading data...")
    start = time.time()
    # Read in anime database: contains all animes along with its associated integer id
    anime_id_map = pd.read_csv('./data/anime_id_title.csv')
    id_2_title = {id: title for id, title in zip(anime_id_map.anime_id, anime_id_map.title)}
    title_2_id = {title: id for id, title in zip(anime_id_map.anime_id, anime_id_map.title)}

    # Read in user rating database: contains all user's and their associated rating to a specific anime
    df = pd.read_csv('./data/user_ratings.csv')
    df = df[['username','anime_id','my_score']]
    username_2_id = {username: id for id, username in enumerate(df.username.unique())}
    id_2_username = {id: username for id, username in enumerate(df.username.unique())}

    # Reduce the size of our dataset:
    # 1. Remove anime that have less than or equal to 5000 ratings.
    # 2. Remove user's that have less than or equal to 10 reviews.
    # 3. Randomly select 5000 users to be part of the recommendation system.

    #anime with less than 10 ratings
    reviews_per_anime = df.groupby('anime_id').count()
    anime_few_reviews = reviews_per_anime[reviews_per_anime.username <= 5000].index.to_list()

    #user's with less than 10 reviews
    review_per_user = df.groupby('username').count()
    users_few_reviews = review_per_user[review_per_user.anime_id <= 10].index.to_list()

    #Remove anime with less than 5000 ratings
    df = df[~df['anime_id'].isin(anime_few_reviews)]

    #Remove user's with less than 10 reviews
    df = df[~df['username'].isin(users_few_reviews)]

    # Map username to integer
    df['username'] = df['username'].map(username_2_id)

    #Randomly sample 5000 users from the dataframe
    index = np.random.randint(low=0, high=len(df.username.unique()), size=5000)
    df = df[df.username.isin(index)]
    df = df.rename(columns={'username':'userID', 'anime_id':'itemID', 'my_score':'rating'})

    # Get data of user's profile, use jikanpy and it's API call to myanimelist.net
    jikan = Jikan()
    user_profile = jikan.user(username=username, request='animelist')
    # Parse data such that it is in the form: (user_id, anime_id, rating)
    user_data = [(len(username_2_id) + 1, title_2_id[anime['title']], anime['score']) 
        for anime in user_profile['anime'] if anime['title'] in title_2_id]

    # Append the new data gotten from API to our original dataset
    tmp = pd.DataFrame(user_data, columns=['userID','itemID','rating'])
    print("old df shape: {}".format(df.shape))
    df = df.append(tmp,ignore_index = True)

    end = time.time()
    print("Time Elapsed: {}".format(end - start))

    print('new df shape: {}'.format(df.shape))
    return df

if __name__ == '__main__':
    prep_data('FreeMakintosh')