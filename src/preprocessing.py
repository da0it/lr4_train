import re
import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from config.params import PREPROCESSING, PATHS

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        if PREPROCESSING['lemmatize']:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
        return ' '.join(tokens)

    def prepare_data(self, texts, labels):
        cleaned_texts = [self.clean_text(t) for t in texts]
        
        self.vectorizer = TfidfVectorizer(
            max_features=PREPROCESSING['max_features'],
            min_df=PREPROCESSING['min_df'],
            max_df=PREPROCESSING['max_df'],
            ngram_range=PREPROCESSING['ngram_range']
        )
        
        X = self.vectorizer.fit_transform(cleaned_texts)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, 
            test_size=PREPROCESSING['test_size'],
            random_state=42,
            stratify=labels
        )
        
        self.save_vectorizer()
        return X_train, X_test, y_train, y_test

    def save_vectorizer(self):
        joblib.dump(self.vectorizer, PATHS['vectorizer'])