from preprocessing import TextPreprocessor
from linear_model import LinearClassifier
from neural_network import NNTrainer
from evaluate import evaluate_linear_model, evaluate_nn, plot_metrics
from config.params import PREPROCESSING, NN
import joblib
import numpy as np
import pandas as pd

# Пример загрузки данных (замените на свои данные)
def load_data():
    df = pd.read_csv('data/twitter_training.csv')
    
    # Фильтрация или дополнительная обработка
    df = df[df['text'].notna()]  # Удаляем пустые тексты
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])]  # Фильтруем только определенные метки
    
    # Дополнительные преобразования при необходимости
    texts = df['text'].values
    labels = df['label'].map({'pos': 1, 'neg': 0}).values  # Преобразование строковых меток в числовые
    
    return texts, labels


def main():
    # 1. Загрузка данных
    texts, labels = load_data()  # Ваши данные с метками Positive/Negative/Neutral
    
    # 2. Подготовка данных
    preprocessor = TextPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(texts, labels)
    
    # 3. Обучение нейросети
    print("\nTraining Neural Network for 3-class classification...")
    nn_trainer = NNTrainer(
        input_size=X_train.shape[1],
        num_classes=len(preprocessor.label_encoder.classes_)
    )
    nn_trainer.train(X_train, y_train, X_test, y_test)
    nn_trainer.save_model()
    
    # 4. Оценка
    evaluate_nn(
        nn_trainer.model, 
        X_test, 
        y_test,
        label_encoder=preprocessor.label_encoder  # Передаем для расшифровки меток
    )

if __name__ == "__main__":
    main()