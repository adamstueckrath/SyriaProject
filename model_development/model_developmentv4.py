import pprint
import pathlib
import numpy as np
import pandas as pd



def read_csv_data(df):
    """
    Reads two csv files (tweets and events) and 
    returns a pandas dataframe for each file
    """
    return pd.read_csv(df, header=0)


def string_to_datetime(t_date):
    """
    Turns a datetime string like this: 
    '2017-07-06T18:34:37.000Z' 
    to a Python datetime object like this -> 2017-07-06 18:34:41
    """
    
    return pd.to_datetime(t_date).date()


def remove_tweets_filter(tweets_df, events_df, filter_amount=None):
    """
    Drops Nan values from the tweet_text_normalize column and 
    returns dataframe filtered or whole
    """
    tweets_df = tweets_df.dropna(subset=['tweet_text_normalize'])
    if filter_amount:
        tweets_df = tweets_df[:filter_amount]

    tweets_df['tweet_created_at'] = tweets_df['tweet_created_at'].apply(string_to_datetime)
    events_df['event_date'] = events_df['event_date'].apply(string_to_datetime)

    return tweets_df, events_df


def eval_dataframe(tweets_df, events_df):
    """
    Evaluates dataframe column saved as string format 
    into a python list object
    """
    tweets_df['tweet_text_normalize'] = tweets_df['tweet_text_normalize'].apply(eval)
    events_df['event_text_normalize'] = events_df['event_text_normalize'].apply(eval)

    return tweets_df, events_df


# set directory path data
syria_data_dir = pathlib.Path('/Users/adamstueckrath/Desktop/syria_data/')

# syria_events_csv file path
events_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'events_pre_processed.csv'

# tweets_no_rts_csv file path
tweets_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'tweets_pre_processed.csv'

results_csv = syria_data_dir / 'model' / 'model_data' / 'tweet_modelv10.csv'

events_df = read_csv_data(events_pre_processed_csv)
tweets_df = read_csv_data(tweets_pre_processed_csv)
results_df = read_csv_data(results_csv)

tweets_df, events_df= remove_tweets_filter(tweets_df, events_df) 
tweets_df, events_df = eval_dataframe(tweets_df, events_df)
print('tweets {} events {}'.format(tweets_df.shape, events_df.shape))

tweets_df_subset = tweets_df.drop(['tweet_id','tweet_lang', 'user_id_str', 
                                       'user_name', 'tweet_text_clean', 'tweet_text_tokenize'], axis=1)
events_df_subset = events_df.drop(['event_type', 'actor_1', 'assoc_actor_1', 
                                       'actor_2', 'assoc_actor_2', 'latitude', 
                                       'longitude', 'event_text_clean', 'event_text_tokenize'], axis=1 )
print('tweets subset {} events subset {}'.format(tweets_df_subset.shape, events_df_subset.shape))

tweets_df_subset = tweets_df_subset.set_index('tweet_id_str')
# events_df_subset = events_df_subset.set_index('event_id')
cosine_results_df_tweet_idx = results_df.set_index('tweet_id_str')
# cosine_results_df_tweet_idx = cosine_results_df.set_index('event_id')

results_df = tweets_df_subset.join(cosine_results_df_tweet_idx)
results_df.head()


