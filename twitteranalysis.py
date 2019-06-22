#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier
import tweepy
import csv
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 


warnings.filterwarnings("ignore", category=DeprecationWarning)

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train  = pd.read_csv('data/training.1600000.processed.noemoticon.csv',encoding = "ISO-8859-1",names=['target','ids','date','flag','user','text'])


# In[3]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt  


# In[4]:


train['clean_tweet']=np.vectorize(remove_pattern)(train['text'], "@[\w]*")


# In[5]:


train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[6]:


train['clean_tweet'] = train['clean_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[7]:


def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data


# In[8]:


training_data, evaluation_data = train_test_split(train, test_size=0.2)


# In[9]:


training_data_selected = training_data.drop(['ids','date','flag','user','text'],axis=1)


# In[10]:


#evaluation_data_selected = evaluation_data.drop(['ids','date','flag','user','text'],axis=1)


# In[11]:


#training_data_selected_dict = dict(zip(training_data_selected.clean_tweet, training_data_selected.target))


# In[12]:


evaluation_data_selected = evaluation_data.drop(['ids','date','flag','user','text'],axis=1)
evaluation_data_selected.to_csv('data/training.1600000.processed.noemoticon_test.csv', encoding='utf-8', index=False, columns=["clean_tweet","target"])


# In[13]:


training_data_selected.to_csv('data/training.1600000.processed.noemoticon_train.csv', encoding='utf-8', index=False, columns=["clean_tweet","target"])


# In[14]:


with open('data/training.1600000.processed.noemoticon_train.csv', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="csv")


# In[ ]:


with open('data/training.1600000.processed.noemoticon_test.csv', 'r') as fp:
    print(cl.accuracy(fp))


# In[16]:


consumer_key = "bQTxFUG99KfRrATIE0OncIq0J"
consumer_secret = "TvR3gJ7y1YZ6Or9KkiDMVxp7gFIkM0j7k3I480Gipivw7KsX4H"
access_token = "3303138865-gqhgjAmeQ6LHywdPJUwCuBA08Y2ZN8W46T7KOHW"
access_token_secret = "9o1YtfOm1Gt89k0hQhQ1Mx0YKKyS0JPWE5CgE8zmJXLOB"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#public_tweets = api.home_timeline()
public_tweets = api.search(input('Enter keyword you want to search on Twitter:'))

words_neg = ''
words_pos = ''
for tweets in public_tweets:
    tweet = tweets.text.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    analysis = cl.classify(tweet)
    if (analysis == 0):
        words_neg += tweet
    else:
        words_pos += tweet


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
wordcloud_neg = WordCloud(stopwords = 
                      STOPWORDS, background_color='black', 
                      height = 2500, width = 3000).generate(words_neg)
wordcloud_pos = WordCloud(stopwords = 
                      STOPWORDS, background_color='black', 
                      height = 2500, width = 3000).generate(words_pos)


# In[15]:


plt.imshow(wordcloud_neg)
plt.imshow(wordcloud_pos)
plt.axis('off')
wordcloud_neg.to_file("img/negative_tweets.png")
wordcloud_pos.to_file("img/positive_tweets.png")

