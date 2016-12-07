from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import tokenize
import os
import json

sid = SentimentIntensityAnalyzer()

ss = sid.polarity_scores('hello, I am not fine')

os.chdir("..")

inputFile = 'clustered_tweets_with_emotion_doc2vec.json'
outputFile = 'clustered_tweets_with_emotion_doc2vec_pos_neg.json'

outFile = open(outputFile, 'w')
with open(inputFile) as f:
	for line in f:
		tweet = json.loads(line)
		
		
	
		tweetText = json.dumps(tweet['tweet_text'])
		tweetTextFiltered = ''

		for c in tweetText:
			if c.isalnum() or c == ' ':
				tweetTextFiltered += c
		
		polarity = sid.polarity_scores(tweetTextFiltered)
		tweet['sentiment'] = polarity
		outFile.write("%s\n" % json.dumps(tweet))




	



