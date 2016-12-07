import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from gensim import corpora, models
import gensim

os.chdir("..")

fname = 'ldaModel.lda'
model = gensim.models.ldamodel.LdaModel.load(fname)


print(model.print_topics(num_topics=20, num_words=4))







