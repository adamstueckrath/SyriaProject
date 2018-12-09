# import core libraries 
import csv
import ast
import pprint
import pathlib
import itertools
import time
import itertools

# import third-party libraries
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
from concurrent.futures import as_completed, ThreadPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


__version__ = '0.3.2'

def string_to_datetime(t_date):
    """
    Turns a datetime string like this: 
    '2017-07-06T18:34:37.000Z' 
    to a Python datetime object like this -> 2017-07-06 18:34:41
    """
    
    return pd.to_datetime(t_date).date()


def read_csv_data(tweet_csv, event_csv):
    """
    Reads two csv files (tweets and events) and 
    returns a pandas dataframe for each file
    """
    tweets_df = pd.read_csv(tweet_csv, header=0)
    events_df = pd.read_csv(event_csv, header=0)

    return tweets_df, events_df


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


def date_ranginater(tweets_df, events_df):
    """
    Returns a dictionary of date ranges where the key is the event date and
    the value is the tweet dates.
    Example: 
    {datetime.date(2017, 7, 6): [datetime.date(2017, 7, 6),
                                 datetime.date(2017, 7, 7),
                                 datetime.date(2017, 7, 8),
                                 datetime.date(2017, 7, 9)]}
    """
    tweet_dates = sorted(tweets_df.tweet_created_at.unique().tolist())
    event_dates = sorted(events_df.event_date.unique().tolist())
    my_dict = dict()
    for index, e_date in enumerate(event_dates):
        my_dict[e_date] = tweet_dates[index:index + 4]

    return my_dict


def create_corpus(tweets_df, events_df):
    """
    Creates two corpus' for the algorithm. The tweet corpus
    will be the training set and the event corpus will be the query.
    """
    # column to list
    tweet_normalize_list = tweets_df.tweet_text_normalize.tolist()
    event_normalize_list = events_df.event_text_normalize.tolist()
    # join processed text together for corpus'
    tweet_corpus = [ ' '.join(x) for x in tweet_normalize_list ]
    event_corpus = [ ' '.join(x) for x in event_normalize_list ]

    return tweet_corpus, event_corpus


def tfidf_algo(tweet_corpus, event_corpus, print_shape=False):
    """
    returns a two document term matrices of document (y) and featured words (x)

    fit_transform(): Learn vocabulary and idf, return term-document matrix.
    transform(): Transform documents to document-term matrix. Uses the vocabulary and 
    document frequencies (df) learned by fit_transform()
    """
    vectorizer = TfidfVectorizer(min_df=.000025)
    tweetVectorizerArray = vectorizer.fit_transform(tweet_corpus).toarray()
    eventVectorizerArray = vectorizer.transform(event_corpus).toarray()
    if print_shape:
        print('Tweets: {} Events: {}'.format(tweetVectorizerArray.shape, eventVectorizerArray.shape))
    return tweetVectorizerArray, eventVectorizerArray


def cosine_x(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity 
    according to the definition of the dot product
    """
    return round(np.inner(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)), 3)


def query_vectors_cosine(tweetVectorizerArray, eventVectorizerArray, tweet_id_list, event_id_list):
    """
    https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
    """
    tweet_event_cosine = []
    counter_numbers = [ num for num in range(0, len(tweetVectorizerArray), 1000) ]
    counter = 0
    for tweet_id, tweet_vector in zip(tweet_id_list, tweetVectorizerArray):
        cosine_dict = dict()
        counter += 1
        if counter in counter_numbers:
            print('Processed {} out of {} tweets'.format(counter, len(tweetVectorizerArray)))
        for event_id, event_vector in zip(event_id_list, eventVectorizerArray):
            cosine = cosine_x(tweet_vector, event_vector)
            cosine_dict[event_id] = cosine

        event_id_max_cosine = max(cosine_dict, key=cosine_dict.get)
        event_id_max_cosine_value = cosine_dict[event_id_max_cosine]
        tweet_event_cosine.append((tweet_id, event_id_max_cosine, event_id_max_cosine_value))

    return tweet_event_cosine


def main():
    """
    Main control function
    """
    start = time.time() # start timer
    print('Start time: {}'.format(start))

    # set directory path data /Users/adamstueckrath/Desktop/stueckrath.adam/syria_data/ 
    syria_data_dir = pathlib.Path('/Users/adamstueckrath/Desktop/syria_data/')
    tweets_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'tweets_pre_processed.csv'
    events_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'events_pre_processed.csv'

    # process data
    tweets_df, events_df = read_csv_data(tweets_pre_processed_csv, events_pre_processed_csv)
    tweets_df, events_df= remove_tweets_filter(tweets_df, events_df) 
    tweets_df, events_df = eval_dataframe(tweets_df, events_df)
    date_chucks = date_ranginater(tweets_df, events_df)
    cosine_results = []

    # iterate over ranginater chuncks and filter data in dates
    for event_date, tweet_dates in date_chucks.items():
        print(event_date, tweet_dates)
        tweet_mask = (tweets_df['tweet_created_at'] >= tweet_dates[0]) & (tweets_df['tweet_created_at'] <= tweet_dates[-1])
        event_mask = (events_df['event_date'] == event_date)
        tweet_df = tweets_df.loc[tweet_mask]
        event_df = events_df.loc[event_mask]
        print(tweet_df.shape, event_df.shape)
        tweet_corpus, event_corpus = create_corpus(tweet_df, event_df)

        # begin algorithm for date chunck
        tweet_array, event_array = tfidf_algo(tweet_corpus, event_corpus, print_shape=True) 
        tweet_id_list  = tweet_df.tweet_id_str.tolist()
        event_id_list = event_df.event_id.tolist()
        cosine_query_results = query_vectors_cosine(tweet_array, event_array, tweet_id_list, event_id_list)
        
        # append results to cosine_results list
        cosine_results.append(cosine_query_results)

    # write tweets to csv 
    cosine_results = list(itertools.chain.from_iterable(cosine_results))
    cosine_results_df = pd.DataFrame(cosine_results, columns=['tweet_id_str', 'event_id', 'consine_value'])
    cosine_results_csv = syria_data_dir / 'model' / 'model_data' / 'tweet_modelv8.csv'
    cosine_results_df.to_csv(cosine_results_csv, index=False)
    
    end = time.time() # end timer
    print('End time: {}'.format(end))
    print('Total time: {}'.format(end-start))

if __name__ == '__main__':
    main()

