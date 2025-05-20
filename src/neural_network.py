import sys
sys.path.append('../')
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config.params import NN, PATHS
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

class TextClassifierNN(nn.Module):
    def __init__(self, input_size, num_classes, initialization='xavier'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, NN['hidden_size'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(NN.get('dropout_rate', 0.3))
        self.fc2 = nn.Linear(NN['hidden_size'], num_classes)
        self._init_weights(initialization)

    def _init_weights(self, method):
        if method == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif method == 'he':
            nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        else:  # 'zero'
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class NNTrainer:
    def __init__(self, input_size, num_classes):
        self.model = TextClassifierNN(input_size, num_classes, initialization='xavier')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=NN['learning_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Создаем модели с разными инициализациями
        self.models = {
            'zero': self._build_model(input_size, num_classes, 'zero'),
            'xavier': self._build_model(input_size, num_classes, 'xavier'),
            'he': self._build_model(input_size, num_classes, 'he')
        }
        self.history = {init: {'train': [], 'test': []} for init in self.models}

    def _build_model(self, input_size, num_classes, initialization):
        """Создает новую модель TextClassifierNN с указанной инициализацией"""
        model = TextClassifierNN(input_size, num_classes, initialization)
        model.to(self.device)
        return model

    def train(self, X_train, y_train, X_test, y_test):
        X_train_t = torch.tensor(X_train.toarray(), dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test.toarray(), dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        for init, model in self.models.items():
            print(f"\nTraining with {init} initialization...")
            optimizer = optim.Adam(model.parameters(), lr=NN['learning_rate'])
            
            for epoch in range(NN['num_epochs']):
                # Train
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = nn.CrossEntropyLoss()(outputs, y_train_t)
                loss.backward()
                optimizer.step()
                self.history[init]['train'].append(loss.item())
                
                # Test
                model.eval()
                with torch.no_grad():
                    test_loss = nn.CrossEntropyLoss()(model(X_test_t), y_test_t).item()
                    self.history[init]['test'].append(test_loss)
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Test Loss={test_loss:.4f}")

    def plot_training(self):
        plt.figure(figsize=(12, 6))
        for init in self.models:
            plt.plot(self.history[init]['train'], label=f'{init} train')
            plt.plot(self.history[init]['test'], '--', label=f'{init} test')
        plt.title('Training Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def _train_epoch(self, loader):
        self.model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

    def _evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        return correct / total

    def _create_loader(self, X, y, shuffle):
        """Создание DataLoader с проверкой типов данных"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        if isinstance(y[0], str):
            raise ValueError("Метки должны быть числовыми. Используйте LabelEncoder для преобразования")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=NN['batch_size'], shuffle=shuffle)

    def save_model(self):
        torch.save(self.model.state_dict(), PATHS['neural_net'])