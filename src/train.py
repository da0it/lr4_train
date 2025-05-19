from preprocessing import TextPreprocessor
from sklearn.preprocessing import LabelEncoder
from neural_network import NNTrainer
from evaluate import evaluate_nn
from config.params import PREPROCESSING, NN
from pathlib import Path
import pandas as pd
import sys, os, joblib

def load_data():
    """Загрузка данных с учетом особенностей формата"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "twitter_training.csv"
        
        # Чтение с явным указанием формата
        df = pd.read_csv(
            data_path,
            header=None,
            names=['id', 'game', 'sentiment', 'text', 'extra'],  # 5 столбцов
            usecols=['text', 'sentiment'],  # Берем только нужные
            encoding='utf-8',
            quoting=3,
            on_bad_lines='warn',
            engine='python'
        )

        # Удаление возможных пустых столбцов
        df = df.dropna(axis=1, how='all')
        
        # Проверка данных
        print("Первые 5 строк после загрузки:")
        print(df.head())
        
        # Фильтрация
        valid_sentiments = ['Positive', 'Negative', 'Neutral']
        df = df[df['sentiment'].isin(valid_sentiments)]
        df = df.dropna(subset=['text', 'sentiment'])
        df = df[df['text'].str.strip().astype(bool)]
        
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['sentiment'])

        os.makedirs('models', exist_ok=True)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')

        print("\nРаспределение меток:")
        print(df['sentiment'].value_counts())
        
        print("Пример преобразования меток:")
        for original, encoded in zip(df['sentiment'][:5], labels[:5]):
            print(f"{original} -> {encoded}")
        
        return df['text'].values, labels
        
    except Exception as e:
        print(f"Ошибка загрузки: {str(e)}")
        sys.exit(1)

def main():
    # 1. Загрузка данных
    print("Загрузка данных...")
    texts, labels = load_data()

    # Проверка меток
    print(f"Тип меток: {type(labels[0])}, Примеры: {labels[:10]}")
    
    # 2. Подготовка данных
    print("\nПодготовка данных...")
    preprocessor = TextPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(texts, labels)
    
    # 3. Обучение нейросети
    print("\nОбучение нейросети...")
    nn_trainer = NNTrainer(
        input_size=X_train.shape[1],
        num_classes=len(preprocessor.label_encoder.classes_)
    )
    nn_trainer.train(X_train, y_train, X_test, y_test)
    nn_trainer.save_model()
    
    # 4. Оценка
    print("\nОценка модели...")
    evaluate_nn(
        nn_trainer.model, 
        X_test, 
        y_test,
        label_encoder=preprocessor.label_encoder
    )

if __name__ == "__main__":
    main()