import pandas as pd
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('omw-1.4')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

df=pd.read_csv('AmazonAlexa_Reviews.csv')
df

# I. Plot a graph of Positive and Negative Feedback
feedback_counts = df['feedback'].value_counts()
feedback_counts.plot(kind='bar')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.title('Positive and Negative Feedback')
plt.show()

# II. Convert the review text into lowercase
df['verified_reviews'] = df['verified_reviews'].str.lower()
df['verified_reviews']

# III. Remove all punctuations from review text
df['verified_reviews'] = df['verified_reviews'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['verified_reviews']

# IV. Remove emoticons and emojis from the text
def remove_emoticons(text):
    emoticon_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text)

df['verified_reviews'] = df['verified_reviews'].apply(remove_emoticons)
df['verified_reviews']

# V. Tokenize the review text into words
df['verified_reviews'] = df['verified_reviews'].apply(word_tokenize)
df['verified_reviews']


# VI. Create representation of Review Text by calculating Term Frequency and Inverse Document Frequency (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['verified_reviews'].apply(lambda x: ' '.join(x)))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
tfidf_df.head(1)

# VI. Remove the Stopwords from the tokenized text
stop_words = set(stopwords.words('english'))
df['verified_reviews'] = df['verified_reviews'].apply(lambda x: [word for word in x if word not in stop_words])
df['verified_reviews']

# IV. Perform stemming & lemmatization on the review text
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
df['verified_reviews'] = df['verified_reviews'].apply(lambda x: [stemmer.stem(word) for word in x])
df['verified_reviews'] = df['verified_reviews'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['verified_reviews']



# VI. Create representation of Review Text by calculating Term Frequency and Inverse Document Frequency (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['verified_reviews'].apply(lambda x: ' '.join(x)))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
tfidf_df.head(1)