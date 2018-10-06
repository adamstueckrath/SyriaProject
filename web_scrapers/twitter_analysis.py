import pandas
import json
import ijson
import pprint
from itertools import islice
from pandas.io.json import json_normalize
from pandas import ExcelWriter

# read n number of json objects from the tweets dataset
def read_n_from_file(json_path, n_lines):
    data = []
    with open(json_path) as f:
        for line in islice(f, n_lines):
            data.append(json.loads(line))
        return(data)  

# %%timeit
# get the total number of json objects in file 
# json objects must be stored per line, not in an arrary 
# file contains separate JSON object on each line.
def count_json_objects(json_path):
    count = 0
    with open(json_path, 'r') as file:
        for line in file: 
            count+=1
        return count

def count_json_objects_chunk(json_path):
    count = 0
    with open(json_path) as f:
        while True:
            next_n_lines = list(islice(f, 20))
            if not next_n_lines:
                break
            for line in next_n_lines: 
                count += 1
        return count

# get column names and types
def get_columns_types(dataframe):
    column_details = {}
    columns = dataframe.columns.values.tolist()
    for column in columns: 
        column_details[column] = type(dataframe[column].iat[0])
    return column_details

# new line json streamer
def nljson_generator(json_path):
    with open(json_path) as file:
        for line in file: 
            yield json.loads(line)
        
# load json to dataframe
def json_normalize_dataframe(json_object):
    dataframe = json_normalize(json_object)
    return dataframe

# source code: https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
# flattens nested json
def flatten_json(json_object):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(json_object)
    return out


# create sample excel file with n number of tweets 
# read from json file, normalize json, and create dataframe
# transpose dataframe and save each tweet into seperate excel tab
def json_tweets_xlsx_sample(json_path, excel_path, 
                         tweet_start=0, tweet_limit='all', 
                         normalize=False):
    tweet_list = []
    json_stream = nljson_generator(json_path)
    
    if tweet_limit == 'all':
        tweets = json_stream
    else:
        if isinstance(tweet_limit, int): 
            tweets = islice(json_stream, tweet_start, tweet_start+tweet_limit)
    
    for tweet in tweets: 
        if normalize:
            tweet_normalize = json_normalize_dataframe(tweet)
            tweet_object = tweet_normalize.transpose()
        else:
            tweet_object = pandas.Series(tweet)
        tweet_list.append(tweet_object)
            
    print("writing tweets")
    writer = ExcelWriter(excel_path)
    for n, tweet in enumerate(tweet_list):
        tweet.to_excel(writer,'tweet%s' % n)
    writer.save()

def remove_retweets(json_path, output_json_path):
    json_stream = nljson_generator(json_path)
    with open(output_json_path, 'w') as output:
        for tweet in json_stream:
            text = tweet['text']
            if text.startswith('RT'):
                continue
            else:
                json.dump(tweet, output)
                output.write("\n")

def nljson_to_dataframe(json_path):
    json_obj_list = []
    json_stream = nljson_generator(tweets_no_retweets_path)
    for json_obj in json_stream:
        json_obj_list.append(json_obj)
    return json_obj_list 




if __name__ == '__main__':
	
	# set file path for tweets dataset
	tweet_path = '/Users/adamstueckrath/Desktop/SyriaProjectNotes/data/tweets/tweets.json'
	# get sample json objects 
	sample_json_objects = read_n_from_file(tweet_path, 1)
	# print sample
	pprint.pprint(sample_json_objects)

	tweets_no_retweets_path = '/Users/adamstueckrath/Desktop/tweets_no_retweets.json'
	tweets = nljson_to_dataframe(tweets_no_retweets_path)
	print(len(tweets))
