import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd

model = api.load("word2vec-google-news-300") # where is the file?


#read file
dataFile= pd.read_csv('synonyms.csv')

#refernce sentence
ref_sent= dataFile.iloc[0]
print(ref_sent)
#victorize tokens
ref_sent_vec= model(ref_sent)
#run loop to extract all text
all_docs= [model(row) for row in dataFile['text']]

similarities=[] #for every similar sentence similarity score


#for i in range(len(all_docs)):
#    similarities= all_docs[i].similarity(ref_sent_vec)# using similarity method from gensim library
#    similarities.append(similarities) #Append similarity score