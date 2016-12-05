import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer


import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3


import os 
num_clusters = 25
os.chdir("..")
tweets = []
tweetTexts = []
with open('christmas.json') as f:
    for line in f:
    	d = json.loads(line)

    	tweetTexts.append(d['tweet_text'])
    	tweets.append(d)


stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 tokenizer=tokenize_and_stem,
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(tweetTexts) 



km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

thefile = open('clustered_tweets.json', 'w')





for i in range(len(tweets)):
	tweetJson = tweets[i]
	tweetJson['cluster'] = clusters[i]
	thefile.write("%s\n" % json.dumps(tweetJson))
	



