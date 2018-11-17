# import core libraries 
import csv
import ast
import pprint
import pathlib
import itertools

# import third-party libraries
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy.linalg as LA


# set directory path data
syria_data_dir = pathlib.Path('/Users/adamstueckrath/Desktop/syria_data/')

# syria_events_csv file path
events_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'events_pre_processed.csv'

# tweets_no_rts_csv file path
tweets_pre_processed_csv = syria_data_dir / 'model' / 'model_data' / 'tweets_pre_processed.csv'

# read dataframes
tweets_df = pd.read_csv(tweets_pre_processed_csv, header=0)
events_df = pd.read_csv(events_pre_processed_csv, header=0)
tweets_df = tweets_df.dropna(subset=['tweet_text_normalize'])

tweet_subset = tweets_df.copy()
tweet_subset = tweet_subset[:100000]

# create list objects
tweet_subset['tweet_text_normalize'] = tweet_subset['tweet_text_normalize'].apply(eval)
events_df['event_text_normalize'] = events_df['event_text_normalize'].apply(eval)

# set list variables
tweet_normalize_text = tweet_subset.tweet_text_normalize.tolist()
event_normalize_text = events_df.event_text_normalize.tolist()

# join normalized text
tweet_normalize_text = [ ' '.join(x) for x in tweet_normalize_text ]
event_normalize_text = [ ' '.join(x) for x in event_normalize_text ]

# algo 
tweet_set = tweet_normalize_text # tweets
event_set = event_normalize_text # event query

vectorizer = TfidfVectorizer()
tweetVectorizerArray = vectorizer.fit_transform(tweet_set).toarray()
eventVectorizerArray = vectorizer.transform(event_set).toarray()

event_id_list = events_df.event_id.tolist()
tweet_event_ids = []
cosine_x = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

for tweet_vector in tweetVectorizerArray:
	cosine_dict = dict()
	for event_id, event_vector in zip(event_id_list, eventVectorizerArray):
		cosine = cosine_x(tweet_vector, event_vector)
		cosine_dict[event_id] = cosine
	tweet_event_ids.append(max(cosine_dict, key=cosine_dict.get))


# assign event_ids to tweets
tweet_subset['tweet_event_id'] = tweet_event_ids

# tweets_no_rts_csv file path
tweets_model_csv = syria_data_dir / 'model' / 'model_data' / 'tweet_model.csv'

# write tweets to csv 
tweet_subset.to_csv(tweets_model_csv, index=False)



