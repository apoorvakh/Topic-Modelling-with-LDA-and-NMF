from gensim import corpora,models,similarities
#tiny corpus of nine documents
import csv
f = open("India-2013.csv")
csv_f = csv.reader(f,delimiter=",")
the_file = open("dictionary_vector_op(t1).txt",mode="w")
docs = []
documents=[]
for i in csv_f:
    docs.append(i)
    #documents.append(docs[i])
#print documents[0:5]
for i in docs:
    documents.extend(i[j] for j in range(0,len(i)))
from nltk.corpus import stopwords

stoplist = stopwords.words('English')
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

 # remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]
#print(texts)

'''To convert documents to vectors, we will use a document representation called bag-of-words. In this representation,each
document is represented by one vector where each vector element represents a question-answer pair, in the style of:
How many times does the word system appear in the document? Once.
It is advantageous to represent the questions only by their (integer) ids. The mapping between the questions and ids is
called a dictionary
'''
dictionary = corpora.Dictionary(texts)
dictionary.save("dict1(t1).dict")
#the_file.write("\n------------PRINT DICTIONARY--------- \n")
print(dictionary)
the_file.write(' '.join(map(str,dictionary)))
'''
Here we assigned a unique integer id to all words appearing in the corpus with the gensim
.corpora.dictionary.Dictionary class. This sweeps across the texts, collecting word counts and relevant statistics.
 In the end, we see there are twelve distinct words in the processed corpus, which means each document will be
 represented by twelve numbers (ie., by a 12-D vector).
 To see the mapping between words and their ids:
 '''
the_file.write("\n---------TOKENS------------- \n")
#the_file.write(' '.join(map(str,dictionary.token2id)))
print("------------------TOKENS!!!!!!!!-----------")
print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize("dict1(t1).mm",corpus)#store to disk for later use
print("\n------------------printing corpus----------------- \n")
#the_file.write(' '.join(map(str,corpus)))
#print(corpus)

class MyCorpus(object):
    def __iter__(self):
        for line in documents:
            #assuming there is one document per line
            yield dictionary.doc2bow(line.lower().split())


corpus_memory_friendly = MyCorpus()
print(corpus_memory_friendly) #prints just its address in memory

the_file.write("\n---------printing vectors-----------\n")
#to print all the constituent vectors
print("\n------------constituent vectors!!----------------\n")
#for vector in corpus_memory_friendly:
    #the_file.write(' '.join(map(str,vector)))
    #print(vector)

#corpus is memory friendly because at any point there is only one
#line in the RAM at a time



