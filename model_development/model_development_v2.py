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

# start timer
start = time.time()
print('Start time: {}'.format(start))

# set directory path data /Users/adamstueckrath/Desktop/syria_data/ 
#/home/stueckrath.adam/syria_data/
syria_data_dir = pathlib.Path('/home/stueckrath.adam/syria_data/')

# syria_events_csv file path
events_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'events_pre_processed.csv'

# tweets_no_rts_csv file path
tweets_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'tweets_pre_processed.csv'

# read dataframes
tweets_df = pd.read_csv(tweets_pre_processed_csv, header=0)
events_df = pd.read_csv(events_pre_processed_csv, header=0)
tweets_df = tweets_df.dropna(subset=['tweet_text_normalize'])

tweets_df = tweets_df

# create list objects
tweets_df['tweet_text_normalize'] = tweets_df['tweet_text_normalize'].apply(eval)
events_df['event_text_normalize'] = events_df['event_text_normalize'].apply(eval)

# set list variables
tweet_normalize_text = tweets_df.tweet_text_normalize.tolist()
event_normalize_text = events_df.event_text_normalize.tolist()

# join normalized text
tweet_normalize_text = [ ' '.join(x) for x in tweet_normalize_text ]
event_normalize_text = [ ' '.join(x) for x in event_normalize_text ]

# algo 
tweet_vec = tweet_normalize_text # tweets
event_vec = event_normalize_text # event query

tweet_id_list = tweets_df.tweet_id_str.tolist()
event_id_list = events_df.event_id.tolist()


def cosine_x(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity 
    according to the definition of the dot product
    # cosine_x = lambda a, b : round(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)), 3)
    """
    return round(np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b)), 3)

vectorizer = TfidfVectorizer(min_df=.000009, max_df=.95)
tweetVectorizerArray = vectorizer.fit_transform(tweet_vec).toarray()
eventVectorizerArray = vectorizer.transform(event_vec).toarray()
print(tweetVectorizerArray.shape) 
print(eventVectorizerArray.shape)

tweet_event_ids = []
counter_numbers = [ num for num in range(0, len(tweet_vec), 1000) ]
counter = 0
for tweet_vector in tweetVectorizerArray:
    cosine_dict = dict()
    counter += 1
    if counter in counter_numbers:
        print('Processed {} out of {} tweets'.format(counter, len(tweet_vec)))
    for event_id, event_vector in zip(event_id_list, eventVectorizerArray):
        cosine = cosine_x(tweet_vector, event_vector)
        cosine_dict[event_id] = cosine
    maximum = max(cosine_dict, key=cosine_dict.get)
    tweet_event_ids.append((maximum, cosine_dict[maximum]))

# end timer
end = time.time()
print('End time: {}'.format(end))
print('Total time: {}'.format(end-start))


# assign event_ids to tweets
tweets_df['tweet_event_id'] = tweet_event_ids

# tweets_no_rts_csv file path
tweets_model_csv = syria_data_dir / 'model' / 'model_data' / 'tweet_modelv6.csv'

# write tweets to csv 
tweets_df.to_csv(tweets_model_csv, index=False)


