from gensim.models import Doc2Vec
import os
import json
import numpy
import tensorflow as tf
from sklearn.cluster import KMeans
os.chdir("..")
fname = 'christmas.d2v'
model = Doc2Vec.load(fname)

num_clusters = 25

mainDocVectors = []

tweets = []

inputFile = 'emotion_tagged_tweets.json'
outputFile = 'clustered_tweets_with_emotion_doc2vec.json'

with open(inputFile) as f:
	for line in f:
		try:
			tweet = json.loads(line)
		except:
			continue
		tweets.append(tweet)
		tweetText = json.dumps(tweet['tweet_text'])
		tweetID = json.dumps(tweet['id_str'])
		tweetTextFiltered = ''
		for c in tweetText:
			if c.isalnum() or c == ' ':
				tweetTextFiltered += c
		docVector =  model.infer_vector(tweetTextFiltered.split())
		mainDocVectors.append(docVector)

km = KMeans(n_clusters=num_clusters)

km.fit(mainDocVectors)

clusters = km.labels_.tolist()

thefile = open(outputFile, 'w')


for i in range(len(tweets)):
    tweetJson = tweets[i]
    rt = tweetJson['retweet_count']
    ft = tweetJson['favorite_count']
    score =  rt + ft
    tweetJson['score'] = score
    tweetJson['cluster'] = clusters[i]
    thefile.write("%s\n" % json.dumps(tweetJson))
	
