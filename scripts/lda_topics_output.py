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


outputFile = 'lda_topics.txt'

outFile = open(outputFile, 'w')

model = gensim.models.ldamodel.LdaModel.load(fname)

topicDict = {}

for i in range(20):
	topic = model.print_topic(i)
	topicDict[i] = topic

outFile.write(str(topicDict))
 





















