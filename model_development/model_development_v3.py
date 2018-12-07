# import core libraries 
import csv
import ast
import pprint
import pathlib
import itertools
import time

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

def read_csv_data(tweet_csv, event_csv):
    """
    Reads two csv files (tweets and events) and 
    returns a pandas dataframe for each file
    """
    tweets_df = pd.read_csv(tweet_csv, header=0)
    events_df = pd.read_csv(event_csv, header=0)

    return tweets_df, events_df


def remove_tweets_filter(tweets_df, filter_amount=None):
    """
    Drops Nan values from the tweet_text_normalize column and 
    returns dataframe filtered or whole
    """
    tweets_df = tweets_df.dropna(subset=['tweet_text_normalize'])
    if filter_amount:
        tweets_df = tweets_df[:filter_amount]

    return tweets_df


def eval_dataframe(tweets_df, events_df):
    """
    Evaluates dataframe column saved as string format 
    into a python list object
    """
    tweets_df['tweet_text_normalize'] = tweets_df['tweet_text_normalize'].apply(eval)
    events_df['event_text_normalize'] = events_df['event_text_normalize'].apply(eval)

    return tweets_df, events_df


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
    vectorizer = TfidfVectorizer(min_df=.00003, max_df=.95)
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
    return round(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)), 3)


def query_vectors_cosine(tweetVectorizerArray, eventVectorizerArray, event_id_list):
    """
    https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
    """
    tweet_event_cosine = []
    counter_numbers = [ num for num in range(0, len(tweetVectorizerArray), 1000) ]
    counter = 0
    for tweet_vector in tweetVectorizerArray:
        cosine_dict = dict()
        counter += 1
        if counter in counter_numbers:
            print('Processed {} out of {} tweets'.format(counter, len(tweetVectorizerArray)))
        for event_id, event_vector in zip(event_id_list, eventVectorizerArray):
            cosine = cosine_x(tweet_vector, event_vector)
            cosine_dict[event_id] = cosine

        maximum = max(cosine_dict, key=cosine_dict.get)
        tweet_event_cosine.append((maximum, cosine_dict[maximum]))

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
    tweets_df = remove_tweets_filter(tweets_df) 
    tweets_df, events_df = eval_dataframe(tweets_df, events_df)
    tweet_corpus, event_corpus = create_corpus(tweets_df, events_df)

    # begin algorithm
    tweet_array, event_array = tfidf_algo(tweet_corpus, event_corpus, print_shape=True) 
    event_id_list = events_df.event_id.tolist()
    cosine_similarity_results = query_vectors_cosine(tweet_array, event_array, event_id_list)

    # assign results to tweets dataframe
    tweets_df['tweet_event_cosine'] = cosine_similarity_results
    write tweets to csv 
    tweets_model_csv = syria_data_dir / 'model' / 'model_data' / 'tweet_modelv7.csv'
    tweets_df.to_csv(tweets_model_csv, index=False)
    
    end = time.time() # end timer
    print('End time: {}'.format(end))
    print('Total time: {}'.format(end-start))

if __name__ == '__main__':
    main()

