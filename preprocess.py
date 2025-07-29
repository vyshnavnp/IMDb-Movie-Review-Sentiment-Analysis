import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_review(text):
    """
    Cleans a single review:
    - Lowercases
    - Tokenizes
    - Removes punctuation & non-alpha
    - Removes stopwords
    - Lemmatizes
    Returns a cleaned string.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def prepare_df(df):
    """
    Modifies the DataFrame in-place by:
    - Cleaning the 'review' column → 'cleaned_text'
    - Adding 'word_count' and 'review_length'
    """
    df["cleaned_text"] = df["review"].apply(preprocess_review)
    df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    df["review_length"] = df["cleaned_text"].apply(len)
    df.drop(columns="review",inplace=True)
    return df

