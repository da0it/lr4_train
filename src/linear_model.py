import sys
sys.path.append('../')
import config
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from config.params import LINEAR, PATHS

class LinearClassifier:
    def __init__(self):
        self.model = SGDClassifier(**LINEAR)
        self.train_loss = []
        self.test_accuracy = []

    def train(self, X_train, y_train, X_test, y_test, n_epochs=50):
        for epoch in range(n_epochs):
            self.model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)
            
            self.train_loss.append(1 - train_acc)
            self.test_accuracy.append(test_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={1-train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        self.save_model()
        return self.model

    def save_model(self):
        joblib.dump(self.model, PATHS['linear_model'])