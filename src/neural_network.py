import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config.params import NN, PATHS

class TextClassifierNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, NN['hidden_size'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(NN['dropout_rate'])
        self.fc2 = nn.Linear(NN['hidden_size'], num_classes)
        self._init_weights(NN['initialization'])

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class NNTrainer:
    def __init__(self, input_size, num_classes):
        self.model = TextClassifierNN(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=NN['learning_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, X_train, y_train, X_test, y_test):
        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        test_loader = self._create_loader(X_test, y_test, shuffle=False)
        
        for epoch in range(NN['num_epochs']):
            self._train_epoch(train_loader)
            test_acc = self._evaluate(test_loader)
            print(f"Epoch {epoch+1}: Test Acc={test_acc:.4f}")

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
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
        return correct / len(loader.dataset)

    def _create_loader(self, X, y, shuffle):
        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=NN['batch_size'], shuffle=shuffle)

    def save_model(self):
        torch.save(self.model.state_dict(), PATHS['neural_net'])