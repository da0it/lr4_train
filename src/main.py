import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def normalize_text(text, lemmatize=True):
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление специальных символов и цифр
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Токенизация
    tokens = word_tokenize(text)
    
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Лемматизация
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Пример использования
# df['text'] = df['text'].apply(normalize_text)