from gensim import corpora,models,similarities
import numpy as np
import csv
from nltk.corpus import stopwords
dictionary = corpora.Dictionary.load("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\trial2\\dict1(t1)_new.dict")
corpus = corpora.MmCorpus("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\trial2\\dict1(t1)_new.mm")
stoplist = stopwords.words('English')

f = open("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\trial2\\India-2013.csv")
csv_f = csv.reader(f,delimiter=",")
the_file = open("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\trial2\\dictionary_vector_op(t1).txt",mode="w")
out=open("C:\\Users\\Apoorva\\Desktop\\ABCD\\PESIT\\adaM\\ADA_Sem4\\Codes\\trial2\\result1.txt",mode="w")
#docs is a list of just the titles
docs = []
#here documents is the list of lists.each list contains one title
documents=[]
for i in csv_f:
    docs.append(i)

#print documents[0:5]
for i in docs:
    documents.extend(i[j] for j in range(0,len(i)))
the_file = open("output1(t1)_new.txt",mode="w")
''' the transformations are standard python objects
typically initialized by means of a training corpus'''
#The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus:
#this can be a TFIDF model as well. for that just do models.TfidfModel
lda = models.LdaModel(corpus,num_topics=50)
texts = [word.split() for word in documents if word.lower() not in stoplist]
corpus_test = [dictionary.doc2bow(text) for text in texts]

#applying transformation to a whole corpus:
corpus_lda = lda[corpus_test]

#transformations can also be serialized,one on top of another
lda = models.LdaModel(corpus_lda,id2word=dictionary,num_topics=100,update_every=1,passes=3)

#creating double wrapper over the original corpus
corpus_lda = lda[corpus_lda]

print("-----------------ANOTHER---------------------")
for doc in corpus_lda:
    the_file.write(' '.join(map(str,doc)))
    print doc
    out.write(' '.join(map(str,doc)))
lda.save("model_title.lda")
lda = models.LdaModel.load("model_title.lda")
print("-------------PRINTING TOPICS!!!!-------------")
out.write(' '.join(map(str,"-------------PRINTING TOPICS!!!!-------------")))
#print(lda.print_topics(15))

topics_matrix = lda.show_topics(formatted=False,num_words=25)
topics_matrix = np.array(topics_matrix)
print topics_matrix.shape
print topics_matrix
out.write(' '.join(map(str,topics_matrix.shape)))
out.write(' '.join(map(str,topics_matrix)))

#printing the topics in the end
topic_words = topics_matrix[:,:,1]
print "\n topic words \n"
out.write(' '.join(map(str,"\n topic words \n")))
#print topic_words
for i in topic_words:
    print([str(word) for word in i])
    print()
    out.write(' '.join(map(str,([str(word) for word in i]))))
    out.write(' '.join(map(str,"\n")))


