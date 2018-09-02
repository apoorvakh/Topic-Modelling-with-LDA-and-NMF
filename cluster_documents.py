
from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3

import csv
f = open("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\India-2013_abs\\India-2013.csv")
csv_f = csv.reader(f,delimiter=",")
the_file = open("dictionary_vector_op(t1).txt",mode="w")
output_file=open("output1.txt",mode="w")
docs = []
documents=[]
for i in csv_f:
    docs.append(i)
    #documents.append(docs[i])
#print documents[0:5]
for i in docs:
    documents.extend(i[j] for j in range(0,len(i)))

#print documents[0:2]

import csv
f = open("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\India-2013_abs\\India-2013_abs.csv")
csv_f = csv.reader(f,delimiter=",")
the_file_abs = open("dictionary_vector_op(t1).txt",mode="w")
docs_abs= []
documents_abs=[]
for i in csv_f:
    docs_abs.append(i)
    #documents.append(docs[i])
#print documents[0:5]
for i in docs_abs:
    documents_abs.extend(i[j] for j in range(0,len(i)))

#print documents_abs[0][0:2]

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in documents_abs:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

#print totalvocab_stemmed[0:2]
#print totalvocab_tokenized[0:2]

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')



from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=1.0,max_features=200000,
                                 min_df=0.0 ,stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(documents_abs) #fit the vectorizer to abstract

print (tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
a=(tfidf_vectorizer.get_feature_names())

#s=repr(a)
#print s

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

'''
K-means initializes with a pre-determined number of clusters (I chose 5). Each observation is assigned to a cluster (cluster assignment)
so as to minimize the within cluster sum of squares. Next, the mean of the clustered observations is calculated and used as the new cluster centroid.
Then, observations are reassigned to clusters and centroids recalculated in an iterative process until the algorithm reaches convergence.
'''

from sklearn.cluster import KMeans

num_clusters = 10

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

print (clusters[0:5])
topics = {'title':documents,'abstract': documents_abs,'cluster':clusters}
frame = pd.DataFrame(topics,index=[clusters],columns=['title','abstract','cluster'])
frame['cluster'].value_counts()



print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

print()
print()
