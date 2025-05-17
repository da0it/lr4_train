# Параметры предобработки
PREPROCESSING = {
    'max_features': 15000,
    'min_df': 5,
    'max_df': 0.85,
    'ngram_range': (1, 2),
    'lemmatize': True,
    'test_size': 0.25
}

# Параметры линейной модели
LINEAR = {
    'loss': 'log_loss',
    'penalty': 'l2',
    'alpha': 0.0001,
    'max_iter': 1000,
    'random_state': 42,
    'learning_rate': 'optimal'
}

# Параметры нейросети
NN = {
    'hidden_size': 256,
    'learning_rate': 0.001,
    'num_epochs': 30,
    'batch_size': 128,
    'initialization': 'he'  # 'xavier', 'he' или 'zeros'
}

PATHS = {
    'vectorizer': 'models/tfidf_vectorizer.pkl',
    'linear_model': 'models/linear_model.pkl',
    'neural_net': 'models/neural_net.pt'
}