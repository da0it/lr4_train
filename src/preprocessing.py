import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Создание TF-IDF векторайзера с удалением редких слов
vectorizer = TfidfVectorizer(
    max_features=10000,  # ограничение количества признаков
    min_df=5,            # минимальная частота слова
    max_df=0.8,          # максимальная частота слова
    ngram_range=(1, 2)   # униграммы и биграммы
)

# Преобразование текста в векторы
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)