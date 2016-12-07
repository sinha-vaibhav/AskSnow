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
 
corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=10)

ldamodel.save('./ldaModel.lda')






