import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

def evaluate_nn(model, X_test, y_test, label_encoder=None, target_names=None):
    """Оценка модели (работает и с sklearn, и с PyTorch моделями)"""
    # Для sklearn моделей
    if isinstance(model, BaseEstimator):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    # Для PyTorch моделей
    else:
        X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()
        y_proba = torch.softmax(outputs, dim=1).numpy() if outputs is not None else None
    
    # Отчет о классификации
    print("\nClassification Report:")
    if label_encoder is not None and target_names is not None:
        y_test_str = label_encoder.inverse_transform(y_test)
        y_pred_str = label_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_str, y_pred_str, target_names=target_names))
    else:
        print(classification_report(y_test, y_pred))
    
    # Матрица ошибок с текстовыми метками
    plt.figure(figsize=(8, 6))
    if label_encoder is not None and target_names is not None:
        cm = confusion_matrix(y_test_str, y_pred_str)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=target_names,
            yticklabels=target_names,
            cmap='Blues'
        )
    else:
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues'
        )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()