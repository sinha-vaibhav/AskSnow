# AskSnow
Topic Summarization Platform
Given data in the form of tweets or another format, user gets summary about any topic he searches.
Summary generated using clustering, emotion and sentiment recognition.

Web : Django
Clustering : KMeans
Feature Vectors Extracted using Tf-IDF Models
There are scripts to generate document vectors using LDA and Doc2Vec Model as well
Emotion Recognition : Trained on a multi layer Neural Network via Tensorflow using data generated from alchemy API
Sentiment Recognition : Using vader sentiment module in NLTK
