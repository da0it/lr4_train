import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_nn(model, X_test, y_test, label_encoder=None):
    """Оценка модели нейронной сети"""
    # Преобразование данных в тензоры
    X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(X_tensor)
        _, y_pred = torch.max(outputs, 1)
    
    y_pred = y_pred.numpy()
    
    # Отчет о классификации
    print("\nClassification Report:")
    if label_encoder is not None:
        # Проверяем тип классов
        if isinstance(label_encoder.classes_[0], (np.integer, int)):
            # Если классы числовые, создаем текстовые метки
            class_names = [f"Class_{i}" for i in label_encoder.classes_]
            y_test_str = y_test  # Используем числовые метки как есть
            y_pred_str = y_pred
        else:
            # Если классы текстовые, преобразуем
            class_names = label_encoder.classes_
            y_test_str = label_encoder.inverse_transform(y_test)
            y_pred_str = label_encoder.inverse_transform(y_pred)
        
        print(classification_report(
            y_test_str,
            y_pred_str,
            target_names=class_names
        ))
    else:
        print(classification_report(y_test, y_pred))
    
    # Матрица ошибок
    plt.figure(figsize=(8, 6))
    if label_encoder is not None:
        sns.heatmap(
            confusion_matrix(y_test_str, y_pred_str),
            annot=True,
            fmt='d',
            xticklabels=class_names,
            yticklabels=class_names
        )
    else:
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d'
        )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()