#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv('/home/nesma/SemesterII/Neural Networks/Project/multilingual-text-categorization-dataset/data/dataset.csv', sep='\t', header=None).applymap(str)
dataset.columns = ["Language","label","text"]
languagesData=[]
loc = 0
languages = dataset[dataset.columns[0]].unique()
for i in languages:
    name = languages[loc]+"Data" 
    globals()[name] = pd.DataFrame( dataset[dataset.Language == i])
    loc += 1


# In[2]:


len(englishData)
englishData = englishData


# In[3]:


def lower_words(text):
    text = text.str.lower()
    return text

englishData['text'] = lower_words(englishData['text'])
input_str = englishData['text']
print(englishData.head())


# In[5]:



import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
def removes_url(text):
    text = text.apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    return text

def remove_url(text):
    text = text.apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    return text
englishData['text'] = removes_url(englishData['text'])
englishData['text'] = remove_url(englishData['text'])

englishData.head()


# In[6]:


def remove_numbers(text):
    text = text.str.replace('\d+', '')
    return text


englishData['text'] = remove_numbers(englishData['text'])
englishData.head()


# In[7]:


def remove_punctuations(text):
    text = text.str.replace('[^\w\s]','')
    return text


englishData['text'] = remove_punctuations(englishData['text'])
englishData.head()


# In[8]:



def remove_blank_space(col):
    col = col.str.strip()
    col = col.replace('\s+', ' ', regex=True)   
    return col


englishData['text'] = remove_blank_space(englishData.text)
englishData.head()


# In[9]:


import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
def remove_stop_words(text, stop):    
    text.apply(lambda x: [item for item in x if item not in stop])
    return text

stop = stopwords.words('english')
remove_stop_words(englishData['text'],stop)
englishData.head()


# In[10]:


from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


porter_stemmer = PorterStemmer()
englishData['text'] = englishData['text'].apply(stem_sentences)
englishData.head()


# In[11]:


englishData = englishData[["label","text"]]
englishData['label'] = '__label__' + englishData['label'].astype(str)


# In[12]:


englishData.iloc[0:int(len(englishData)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
englishData.iloc[int(len(englishData)*0.8):int(len(englishData)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
englishData.iloc[int(len(englishData)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False);


# In[13]:


englishData.head()


# In[14]:


from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings,DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')


# In[15]:


len(corpus.dev)


# In[16]:


word_embeddings = [WordEmbeddings('glove'),FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]


# In[17]:


document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)


# In[18]:


classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)


# In[19]:


trainer = ModelTrainer(classifier, corpus)


# In[20]:


trainer.train('./', max_epochs=20, patience=1)


# In[ ]:




