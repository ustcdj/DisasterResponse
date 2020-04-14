import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    '''
    Tokenize text

    INPUT
        text (str): text to be tokenized
    OUTPUT
        tokens (list): list of tokens
    '''

    # Extract all the urls from the provided text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder' )

    # didn't remove punctuation, seems to cause error 'list out of range'
    # text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # normalize text
    tokens = word_tokenize(text)

    # didn't remove stop words. slow down the process, also decreases model performance metrics
    # kokens = [w for w in tokens if w not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# create the 1st custom transformer
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# create the 2nd custom transformer
class TextLenghExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).apply(lambda x: len(x))
        return pd.DataFrame(X)
