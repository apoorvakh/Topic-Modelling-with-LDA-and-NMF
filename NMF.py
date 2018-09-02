__author__ = 'Apoorva'

import os, os.path, codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import numpy as np
from nltk.tokenize import RegexpTokenizer
import networkx as nx

dir_data = "title_texts"
# file_paths = [os.path.join(dir_data, fname) for fname in os.listdir(dir_data) if fname.endswith(".txt") ]
# documents = [codecs.open(file_path, 'r', encoding="utf8", errors='ignore').read() for file_path in dir_data]
final_output = open("final_ngrams3.txt",mode="w")



file_paths = [os.path.join(dir_data, fname) for fname in os.listdir(dir_data) if fname.endswith(".txt") ]
documents = [codecs.open(file_path, 'r', encoding="utf8", errors='ignore').read() for file_path in file_paths ]


for i in dir_data:
    documents.append(i)
print "Read %d corpus of documents" % len(documents)
#print documents[0:3]

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, lowercase=True, strip_accents="unicode", use_idf=True, norm="l1",
                        min_df=2)
A = tfidf.fit_transform(documents)

num_terms = len(tfidf.vocabulary_)
terms = [""] * num_terms
for term in tfidf.vocabulary_.keys():
    terms[tfidf.vocabulary_[term]] = term
print "Created document-term matrix of size %d x %d" % (A.shape[0], A.shape[1])

'''print type(A.shape)
print "list shape"
print li_shape'''

model = decomposition.NMF(init="nndsvd", n_components=30, max_iter=100)
W = model.fit_transform(A)
H = model.components_
#print str(A.shape)
print "H[0]"
print "Generated factor W of size %s and factor H of size %s" % ( str(W.shape), str(H.shape) )
#feature_names = tfidf.get_feature_names()
#print feature_names

#print "terms:"
#print terms
#for i in top_indices:
#    print terms[i]
des_data = open("the_des_list.txt", "r")
keyword_list = []
for i in des_data:
    keyword_list.append(i)
keyword_list=[w.lower() for w in keyword_list]
#print keyword_list[1:5]
G = nx.Graph()
print "H: "
print H[0], len(H[0])
#print terms.index("ionic")

for topic_index in range(H.shape[0]):
    '''For H'''
    top_indices = np.argsort(H[topic_index, :])[::-1][0:15]

    term_ranking = [terms[i] for i in top_indices]

    s = ", ".join(term_ranking)
    tokenizer = RegexpTokenizer(r'\w+')

    sSplit=tokenizer.tokenize(s)
    #print "split:",sSplit
    for l in keyword_list:
            flag=1
            for a in l.split():
                if a not in sSplit:
                    flag=0
            if flag==1:
                sSplit.append(l.strip())
                for j in l.split():
                    sSplit.remove(j)
    topic="topic"+str(topic_index)
    G.add_node(topic)
    G.add_nodes_from(sSplit)

    print "Topic %d: %s" % ( topic_index, ', '.join(sSplit))
    print "\n"
    final_output.write("Topic %d: %s" % ( topic_index, ', '.join(sSplit) ))
    final_output.write("\n")
    for i in sSplit:
        G.add_edge(topic,i)
nx.write_weighted_edgelist(G,'trial3.weighted.edgelist')
#print G.nodes()
nx.write_gml(G,"test3.gml")

'''
sSplit=s.split()
print "split:"
print sSplit

des_data = open("the_des_list.txt", "r")
keyword_list = []
for i in des_data:
    keyword_list.append(i)
print keyword_list[1:10]
keyword_list=[w.lower() for w in keyword_list]

for l in keyword_list:
            flag=1
            for a in l.split():
                if a not in sSplit:
                    flag=0
            if flag==1:
                sSplit.append(l)
                for j in l.split():
                    sSplit.remove(j)

'''



