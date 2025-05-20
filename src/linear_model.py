import sys
sys.path.append('../')
import config
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from config.params import LINEAR, PATHS
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score

class LinearClassifier:
    def __init__(self):
        self.model = SGDClassifier(**LINEAR)
        self.test_loss = []
        self.train_loss = []
        self.train_acc = []
        self.test_acc = []

    def train(self, X_train, y_train, X_test, y_test, penalty='l2', n_epochs=50):
        params = LINEAR.copy()
        params['penalty'] = penalty
        self.model = SGDClassifier(**params)
        
        for epoch in range(n_epochs):
            self.model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            
            train_loss = log_loss(y_train, self.model.predict_proba(X_train))
            test_loss = log_loss(y_test, self.model.predict_proba(X_test))
            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            test_acc = accuracy_score(y_test, self.model.predict(X_test))
            
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
        
        self.save_model()
        return self.model
    
    def plot_weights(self, feature_names, target_names=None, top_n=20):
        """Визуализация важности признаков с текстовыми метками"""
        plt.figure(figsize=(10, 6))
        
        # Используем текстовые метки если они есть, иначе числовые
        class_labels = target_names if target_names is not None else [f'Class {i}' for i in self.model.classes_]
        
        for i, class_name in enumerate(self.model.classes_):
            weights = self.model.coef_[i]
            top_indices = np.argsort(np.abs(weights))[-top_n:]
            plt.barh(
                [feature_names[idx] for idx in top_indices], 
                weights[top_indices], 
                label=class_labels[i]
            )
        
        plt.title('Top Feature Weights by Class')
        plt.xlabel('Weight Value')
        plt.ylabel('Features')
        plt.legend(title='Sentiment Class')
        plt.tight_layout()
        plt.show()
    
    def plot_training(self, target_names=None):
        """График обучения с улучшенным оформлением"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_loss, label='Train', color='blue', linewidth=2)
        ax1.plot(self.test_loss, label='Test', color='orange', linewidth=2)
        ax1.set_title('Training History - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Log Loss')
        ax1.legend(title='Dataset')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Accuracy plot
        ax2.plot(self.train_acc, label='Train', color='blue', linewidth=2)
        ax2.plot(self.test_acc, label='Test', color='orange', linewidth=2)
        ax2.set_title('Training History - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend(title='Dataset')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def save_model(self):
        joblib.dump(self.model, PATHS['linear_model'])