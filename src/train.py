from preprocessing import TextPreprocessor
from sklearn.preprocessing import LabelEncoder
from neural_network import NNTrainer
from linear_model import LinearClassifier
from evaluate import evaluate_nn
from config.params import PREPROCESSING, NN
from pathlib import Path
import pandas as pd
import sys
import os
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_data():
    try:
        data_path = Path(__file__).parent.parent / "data" / "twitter_training.csv"
        df = pd.read_csv(
            data_path,
            header=None,
            names=['id', 'game', 'sentiment', 'text', 'extra'],
            usecols=['text', 'sentiment'],
            encoding='utf-8',
            quoting=3,
            on_bad_lines='warn',
            engine='python'
        )
        df = df.dropna(axis=1, how='all')
        valid_sentiments = ['Positive', 'Negative', 'Neutral']
        df = df[df['sentiment'].isin(valid_sentiments)]
        df = df.dropna(subset=['text', 'sentiment'])
        df = df[df['text'].str.strip().astype(bool)]

        # Сохранение строковых имен классов
        class_names = sorted(valid_sentiments)  # ['Negative', 'Neutral', 'Positive']
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['sentiment'])

        os.makedirs('models', exist_ok=True)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        joblib.dump(class_names, 'models/class_names.pkl')  # Сохраняем имена классов

        return df['text'].values, labels, class_names
    except Exception as e:
        print(f"Ошибка загрузки: {str(e)}")
        sys.exit(1)

def main():
    print("Загрузка данных...")
    texts, labels, class_names = load_data()

    print("\nПодготовка данных...")
    preprocessor = TextPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(texts, labels)
    
    # Получаем имена признаков ДО их использования
    feature_names = preprocessor.vectorizer.get_feature_names_out()
    
    # Линейная модель L2
    print("\n=== Логистическая регрессия (L2) ===")
    linear_l2 = LinearClassifier()
    linear_l2.train(X_train, y_train, X_test, y_test, penalty='l2')
    linear_l2.plot_training()
    linear_l2.plot_weights(feature_names, target_names=class_names)  # Теперь feature_names определен
    
    print("\nОценка модели L2:")
    y_pred_l2 = linear_l2.model.predict(X_test)
    print(classification_report(y_test, y_pred_l2, target_names=class_names))
    
    # Линейная модель L1
    print("\n=== Логистическая регрессия (L1) ===")
    linear_l1 = LinearClassifier()
    linear_l1.train(X_train, y_train, X_test, y_test, penalty='l1')
    linear_l1.plot_training()
    print(f"\nЗануленных весов в L1: {np.sum(linear_l1.model.coef_ == 0)}")
    print(f"Зануленных весов в L2: {np.sum(linear_l2.model.coef_ == 0)}")
    
    print("\nОценка модели L1:")
    y_pred_l1 = linear_l1.model.predict(X_test)
    print(classification_report(y_test, y_pred_l1, target_names=class_names))
    
    # Нейронная сеть
    print("\n=== Нейронная сеть ===")
    nn_trainer = NNTrainer(
        input_size=X_train.shape[1],
        num_classes=len(class_names)
    )
    nn_trainer.train(X_train, y_train, X_test, y_test)
    nn_trainer.plot_training()
    nn_trainer.save_model()
    
    print("\nОценка модели нейронной сети (Xavier инициализация):")
    evaluate_nn(
            nn_trainer.models['xavier'], 
            X_test, 
            y_test,
            label_encoder=preprocessor.label_encoder,
            target_names=class_names
    )
    
    # Сравнение моделей
    print("\n=== Итоговое сравнение моделей ===")
    models = {
        'Logistic (L2)': linear_l2.model,
        'Logistic (L1)': linear_l1.model,
        'Neural Network (xavier)': nn_trainer.models['xavier']
    }
    
    for name, model in models.items():
        if 'Neural' in name:
            with torch.no_grad():
                X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
                if next(model.parameters()).is_cuda:
                    X_tensor = X_tensor.cuda()
                outputs = model(X_tensor)
                y_pred = torch.max(outputs, 1)[1].cpu().numpy()
        else:
            y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        print(f"{name}:")
        print(f"  Accuracy = {acc:.4f}")
        print(f"  F1 Macro = {f1_macro:.4f}")
        print(f"  F1 Micro = {f1_micro:.4f}")

if __name__ == "__main__":
    main()