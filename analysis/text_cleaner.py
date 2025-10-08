"""
Shared text preprocessing utilities for all analysis scripts
"""
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK components
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Stopwords configuration
stop_words = set(stopwords.words("english"))
extra_stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "dont", "ill", "im"
}

STOPWORDS = stop_words.union(extra_stopwords)
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    '''
    Clean and preprocess text for NLP analysis
    '''
    if pd.isna(text):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|t\.co\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize and remove stopwords
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token and token not in STOPWORDS
    ]

    return " ".join(cleaned_tokens)
