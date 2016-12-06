from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import json
import numpy
from random import shuffle

import logging
import os.path
import sys
import cPickle as pickle

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):

        with open(self.filename) as f:
            for line in f:
                tweet = json.loads(line)
                tweetText = tweet['tweet_text']
                tweetID = json.dumps(tweet['id_str'])
                tweetText = json.dumps(tweetText)
                tweetTextFiltered = ''
                for c in tweetText:
                    if c.isalnum() or c == ' ':
                        tweetTextFiltered += c
                yield LabeledSentence(tweetTextFiltered.split(),tags=tweetID)

    def to_array(self):
        self.sentences = []
        with open(self.filename) as f:
            for line in f:
                tweet = json.loads(line)
                tweetText = tweet['tweet_text']
                tweetID = json.dumps(tweet['id_str'])
                tweetText = json.dumps(tweetText)
                tweetTextFiltered = ''
                for c in tweetText:
                    if c.isalnum() or c == ' ':
                        tweetTextFiltered += c
                self.sentences.append(LabeledSentence(words=tweetTextFiltered.split(),tags=tweetID))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
            

os.chdir("..")

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

sentences = LabeledLineSentence('ChristmasClean.json')

model.build_vocab(sentences.to_array())

for epoch in range(5):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

model.save('./christmas_2.d2v')
