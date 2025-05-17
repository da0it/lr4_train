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
    df = df[df['label'].isin(['pos', 'neg'])]  # Фильтруем только определенные метки
    
    # Дополнительные преобразования при необходимости
    texts = df['text'].values
    labels = df['label'].map({'pos': 1, 'neg': 0}).values  # Преобразование строковых меток в числовые
    
    return texts, labels


def main():
    # 1. Загрузка и подготовка данных
    texts, labels = load_data()
    preprocessor = TextPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(texts, labels)
    
    # 2. Обучение линейной модели
    print("\nTraining Linear Model...")
    linear_model = LinearClassifier()
    linear_model.train(X_train, y_train, X_test, y_test)
    evaluate_linear_model(linear_model.model, X_test, y_test)
    plot_metrics(linear_model.train_loss, 
                [1-acc for acc in linear_model.test_accuracy], 
                "Loss")
    
    # 3. Обучение нейросети
    print("\nTraining Neural Network...")
    nn_trainer = NNTrainer(X_train.shape[1], len(np.unique(labels)))
    nn_trainer.train(X_train, y_train, X_test, y_test)
    nn_trainer.save_model()
    
    # Оценка нейросети
    evaluate_nn(nn_trainer.model, X_test, y_test)

if __name__ == "__main__":
    main()