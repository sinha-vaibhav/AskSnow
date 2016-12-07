import os
import json
import numpy
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
os.chdir("..")

fname = 'ldaModel.lda'
#fname = 'ldaModel.d2v'

inputFile = 'clustered_tweets_with_emotion_doc2vec_pos_neg.json'
outputFile = 'clustered_tweets_with_emotion_lda_pos_neg.json'

outFile = open(outputFile, 'w')

model = gensim.models.ldamodel.LdaModel.load(fname)

def GetDocumentVectors(fileName):
	i=1
	docVectors = []
	
	with open(fileName) as f:
		for line in f:
			try:
				tweet = json.loads(line)
			except:
				continue
			tweetText = json.dumps(tweet['tweet_text'])
			tweetID = json.dumps(tweet['id_str'])
			tweetTextFiltered = ''
			for c in tweetText:
				if c.isalnum() or c == ' ':
					tweetTextFiltered += c
			docVectors.append(tweetTextFiltered)		
	return docVectors

docVectors=GetDocumentVectors('ChristmasClean.json')

tokenizer = RegexpTokenizer(r'\w+')


en_stop = get_stop_words('en')


p_stemmer = PorterStemmer()

texts = []


for i in docVectors:
    
 
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

   
    stopped_tokens = [i for i in tokens if not i in en_stop]
    

    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
  
    
    texts.append(stemmed_tokens)


dictionary = corpora.Dictionary(texts)
with open(inputFile) as f:
		i=0
		for line in f:
			try:
				tweet = json.loads(line)
			except:
				continue
			topics = model[dictionary.doc2bow(texts[i])]
			i += 1
			maxTopicProb = 0
			clusterNo = 0
			for topic in topics:
				topicNo = topic[0]
				topicProb = topic[1]
				if maxTopicProb < topicProb:
					clusterNo = topicNo
					maxTopicProb = topicProb
			tweet['cluster'] = clusterNo
			outFile.write("%s\n" % json.dumps(tweet))


















