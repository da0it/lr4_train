import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

class TextClassifierNN(nn.Module):
    def __init__(self, input_size, num_classes, initialization='xavier'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        self._init_weights(initialization)

    def _init_weights(self, method):
        if method == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif method == 'he':
            nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class NNTrainer:
    def __init__(self, input_size, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {
            'zero': TextClassifierNN(input_size, num_classes, 'zero').to(self.device),
            'xavier': TextClassifierNN(input_size, num_classes, 'xavier').to(self.device),
            'he': TextClassifierNN(input_size, num_classes, 'he').to(self.device)
        }
        self.history = {init: {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []} 
                       for init in self.models}
        
    def train(self, X_train, y_train, X_test, y_test, num_epochs=50, batch_size=64, lr=0.001):
        # Преобразование данных в тензоры
        train_dataset = TensorDataset(
            torch.tensor(X_train.toarray(), dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test.toarray(), dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        for init, model in self.models.items():
            print(f"\nTraining with {init} initialization...")
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss, correct, total = 0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                # Evaluation phase
                test_loss, test_correct, test_total = 0, 0, 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        test_loss += criterion(outputs, labels).item()
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()
                
                # Save metrics
                self.history[init]['train_loss'].append(train_loss/len(train_loader))
                self.history[init]['test_loss'].append(test_loss/len(test_loader))
                self.history[init]['train_acc'].append(correct/total)
                self.history[init]['test_acc'].append(test_correct/test_total)
                
                if epoch % 5 == 0 or epoch == num_epochs-1:
                    print(f"Epoch {epoch}: "
                          f"Train Loss={self.history[init]['train_loss'][-1]:.4f}, "
                          f"Test Loss={self.history[init]['test_loss'][-1]:.4f}, "
                          f"Train Acc={self.history[init]['train_acc'][-1]:.4f}, "
                          f"Test Acc={self.history[init]['test_acc'][-1]:.4f}")

    def plot_training(self):
        plt.figure(figsize=(12, 8))
        for init in self.models:
            plt.plot(self.history[init]['train_loss'], label=f'{init} train')
            plt.plot(self.history[init]['test_loss'], '--', label=f'{init} test')
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        for init in self.models:
            plt.plot(self.history[init]['train_acc'], label=f'{init} train')
            plt.plot(self.history[init]['test_acc'], '--', label=f'{init} test')
        plt.title('Training Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()