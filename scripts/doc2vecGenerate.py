from gensim.models import Doc2Vec
import os
import json
import numpy
import tensorflow as tf
os.chdir("..")

fname = 'christmas.d2v'
model = Doc2Vec.load(fname)

def GetDocumentVectors(fileName):
	i=1
	docVectors = []
	emo = []
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
			docVector =  model.infer_vector(tweetTextFiltered.split())

			anger = tweet["emotions"]["anger"]
			joy = tweet["emotions"]["joy"]
			fear = tweet["emotions"]["fear"]
			sadness = tweet["emotions"]["sadness"]
			disgust = tweet["emotions"]["disgust"]

			emotions = numpy.array([anger,joy,fear,sadness,disgust])
			emoOneHot = numpy.zeros(shape=(5))
			emoOneHot[numpy.argmax(emotions)] = 1
			emo.append(emoOneHot)
			docVectors.append(docVector)		
	return numpy.array(docVectors),numpy.array(emo)

docVectors,emo=GetDocumentVectors('training_5k_emotions.json')



testDocs = docVectors[4000:,:]
testEmos = emo[4000:,:]

docVectors = docVectors[:4000,:]
emo = emo[:4000,:]

learning_rate = 0.01
training_epochs = 100
display_step = 1

n_hidden_1 = 256 
n_hidden_2 = 256 
n_input = 100  
n_classes = 5 


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, biases):
   
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    
    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(1):
            batch_x, batch_y = docVectors,emo
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
    
            avg_cost += c 
       
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

  
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Train Accuracy:", accuracy.eval({x: docVectors, y: emo}))
    print("Test Accuracy:", accuracy.eval({x: testDocs, y: testEmos}))


emotion = {0:'anger',1:'joy',2:'fear',3:'sadness',4:'disgust'}
init = tf.initialize_all_variables()
mainDocVectors = []

readFile = 'clustered_tweets.json'
outFile = open('emotion_tagged_tweets_clustered_tfidf.json', 'w')

with open(readFile) as f:
	for line in f:
		tweet = json.loads(line)
		tweetText = json.dumps(tweet['tweet_text'])
		tweetID = json.dumps(tweet['id_str'])
		tweetTextFiltered = ''
		for c in tweetText:
			if c.isalnum() or c == ' ':
				tweetTextFiltered += c
		docVector =  model.infer_vector(tweetTextFiltered.split())
		mainDocVectors.append(docVector)

print "Training Done, Now writing the result File"
with tf.Session() as sess:
	sess.run(init)
	feed_dict = {x: mainDocVectors}
	classification = sess.run(multilayer_perceptron(x, weights, biases), feed_dict)
	i=0
	with open(readFile) as f:
		for line in f:
			tweet = json.loads(line)
			result = numpy.argmax(numpy.array(classification[i]))
			i+=1
			tweet["emotion"] = emotion[result]
			outFile.write("%s\n" % json.dumps(tweet))















