from preprocessing import TextPreprocessor
from neural_network import NNTrainer
from evaluate import evaluate_nn
from config.params import PREPROCESSING, NN
import pandas as pd
from pathlib import Path
import sys

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
        
        print("\nРаспределение меток:")
        print(df['sentiment'].value_counts())
        
        return df['text'].values, df['sentiment'].values
        
    except Exception as e:
        print(f"Ошибка загрузки: {str(e)}")
        sys.exit(1)

def main():
    # 1. Загрузка данных
    print("Загрузка данных...")
    texts, labels = load_data()
    
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
        preprocessor.label_encoder
    )

if __name__ == "__main__":
    main()