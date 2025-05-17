from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import joblib
import torch
import numpy as np

def plot_metrics(train_metrics, test_metrics, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(test_metrics, label=f'Test {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} during training')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_linear_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Linear Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def evaluate_nn(model, X_test, y_test):
    X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, y_pred = torch.max(outputs, 1)
    
    print("Neural Network Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred.numpy()):.4f}")
    print(f"F1 Macro: {f1_score(y_test, y_pred.numpy(), average='macro'):.4f}")