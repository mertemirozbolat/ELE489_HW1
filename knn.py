# knn.py

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

class KNN:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def _distance(self, x1, x2):
        if self.distance_metric == "euclidean":
            return euclidean_distance(x1, x2)
        elif self.distance_metric == "manhattan":
            return manhattan_distance(x1, x2)
        elif self.distance_metric == "minkowski":
            return minkowski_distance(x1, x2, p=3)
        else:
            raise ValueError("Unknown distance metric")

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest).most_common(1)[0][0]
