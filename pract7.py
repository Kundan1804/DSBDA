import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('AmazonAlexa_Reviews.csv')
df.head()

reviews=df.drop(['rating','date','variation','feedback'], axis=1)
reviews_lc = reviews.apply(lambda x: x.astype(str).str.lower())
reviews_lc.head()
df['new reviews']=reviews_lc
def rem_punc(text):
  punc_free = ''.join([i for i in text if i not in string.punctuation])
  return punc_free
df['cleaned reviews']=df['new reviews'].apply(lambda text: rem_punc(text))
df['cleaned reviews'].head()

def tokenize(text):
  tokens = re.split('W+', text)
  return tokens

df['reviews tokenized']=df['cleaned reviews'].apply(lambda x: tokenize(x))
df['reviews tokenized'].head()

nltk.download('stopwords')
', '.join(stopwords.words('english'))

sw = set(stopwords.words('english'))
def rem_sw(text):
  sw_free = ' '.join([word for word in str(text).split() if word not in sw])
  return sw_free
df['wo stop']=df['reviews tokenized'].apply(lambda x: rem_sw(x))
df['wo stop'].head()

stemmer = PorterStemmer()
def stem_words(text):
  stemmed_text = ' '.join([stemmer.stem(word) for word in text.split()])
  return stemmed_text
df['stemmed review']=df['wo stop'].apply(lambda x: stem_words(x))
df['stemmed review'].head()

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemm_words(text):
  return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

df['lemmatized review']=df['wo stop'].apply(lambda x: lemm_words(x))
df['lemmatized review'].head()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = 'This is an example sentence for pos'
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(tagged)

def process_content():
  for i in df['wo stop']:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    return tagged

process_content()

def rem_url(text):
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  return url_pattern.sub(r'', text)
df['url removed']=df['wo stop'].apply(lambda x: rem_url(x))
df['url removed'].head()

tfidf = TfidfVectorizer()
data_tf=tfidf.fit_transform(df['wo stop'])
data_tf

print(data_tf)