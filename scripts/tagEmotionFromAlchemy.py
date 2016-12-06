import json
from watson_developer_cloud import AlchemyLanguageV1
import os
import sys
os.chdir("..")
alchemy_language = AlchemyLanguageV1(api_key='0c65133c4649014a24595522f578898548e1b057')


tweets = []
tweetTexts = []
i = 4700
lower = 4700
higher = 5700

fileName = 'emotionTagged_4000-5000.json'
thefile = open(fileName, 'w')
docCount = lower
with open('ChristmasClean.json') as f:
    for line in f:
    	docCount += 1
    	if(i>=lower):
    		try:

	    		print i
		    	d = json.loads(line)

		    	tweetText = d['tweet_text']
		    	emotion = alchemy_language.emotion(text=tweetText)
		    	d['emotions']= emotion['docEmotions']
		    	i += 1

		    	thefile.write("%s\n" % json.dumps(d))

		except:
			print "Unexpected error:", sys.exc_info()
			continue
	    	
    	if(i>higher):
    		break

print "Final Document Count = ",docCount


