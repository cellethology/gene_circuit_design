#!/usr/bin/env python3

import pickle
from concurrent.futures import ProcessPoolExecutor


def train_and_save_model(config):
    model_name, model, data_path, params = config

    # Load data in each process
    X_train, y_train = load_data(data_path)

    # Train with specific parameters
    model.set_params(**params)
    model.fit(X_train, y_train)

    return model_name, model.score(X_train, y_train)


# Usage
configs = [
    ("rf", RandomForestClassifier(), "data.csv", {"n_estimators": 100}),
    ("svm", SVC(), "data.csv", {"C": 1.0, "kernel": "rbf"}),
    ("lr", LogisticRegression(), "data.csv", {"C": 1.0}),
]

with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(train_and_save_model, config) for config in configs]
