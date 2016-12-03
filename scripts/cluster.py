import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
import os 
num_clusters = 5
os.chdir("..")
tweets = []
tweetTexts = []
with open('christmas.json') as f:
    for line in f:
    	d = json.loads(line)

    	tweetTexts.append(d['tweet_text'])
    	tweets.append(d)

tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000,
                                 min_df=0.01, stop_words='english',
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
	



