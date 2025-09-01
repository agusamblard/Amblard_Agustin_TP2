import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from collections import Counter
import random


class LDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        overall_mean = np.mean(X, axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        self.class_means = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            class_mean = np.mean(X_c, axis=0)
            self.class_means[cls] = class_mean
            self.priors[cls] = X_c.shape[0] / X.shape[0]

            S_W += (X_c - class_mean).T @ (X_c - class_mean)
            mean_diff = (class_mean - overall_mean).reshape(-1, 1)
            S_B += X_c.shape[0] * (mean_diff @ mean_diff.T)

        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
        sorted_indices = np.argsort(-eigvals.real)
        self.W = eigvecs[:, sorted_indices[:len(self.classes)-1]].real
        
        self.cov = S_W / (len(y) - len(self.classes))
        self.inv_cov = np.linalg.pinv(self.cov)

    def predict(self, X):
        X_proj = X @ self.W
        class_projections = {cls: self.class_means[cls] @ self.W for cls in self.classes}
        
        predictions = []
        for x in X_proj:
            distances = {cls: np.linalg.norm(x - proj) for cls, proj in class_projections.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Calcular densidades normales multivariadas para cada clase
        probs = []
        for cls in self.classes:
            mean = self.class_means[cls]
            prior = self.priors[cls]

            # Calcular la probabilidad usando la fórmula de LDA (log-verosimilitud)
            diff = X - mean
            mahalanobis = np.sum(diff @ self.inv_cov * diff, axis=1)
            log_prob = -0.5 * mahalanobis + np.log(prior)
            probs.append(log_prob)

        # Convertir de log-verosimilitud a probabilidad normalizada
        log_probs = np.vstack(probs).T  # shape (n_samples, n_classes)
        probs_exp = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        probs_normalized = probs_exp / np.sum(probs_exp, axis=1, keepdims=True)
        return probs_normalized



def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def split_dataset(X, y, feature, threshold):
    left_indices = X[:, feature] <= threshold
    right_indices = ~left_indices
    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]


def best_split(X, y, num_thresholds=10):
    best_feature, best_threshold, best_gain = None, None, -1
    current_entropy = entropy(y)
    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.percentile(X[:, feature], np.linspace(0, 100, num=num_thresholds))
        thresholds = np.unique(thresholds)  # Evitar repetidos
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            p_left = len(y_left) / len(y)
            p_right = 1 - p_left
            gain = current_entropy - (p_left * entropy(y_left) + p_right * entropy(y_right))
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return np.bincount(y).argmax()

        feature, threshold = best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()

        left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def _predict_one(self, x, node):
        while isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _sample_features(self, X):
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            return random.sample(range(n_features), max(1, int(np.sqrt(n_features))))
        elif isinstance(self.max_features, int):
            return random.sample(range(n_features), min(self.max_features, n_features))
        else:
            return list(range(n_features))

    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y))
        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            feature_indices = self._sample_features(X_sample)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        predictions = np.array([
            tree.predict(X[:, features])
            for tree, features in self.trees
        ])
        return np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        y_scores = np.zeros((n_samples, n_classes))

        for tree, feature_indices in self.trees:
            preds = tree.predict(X[:, feature_indices])
            for i, pred in enumerate(preds):
                class_idx = self.classes_.index(pred)
                y_scores[i, class_idx] += 1

        y_scores /= self.n_estimators
        return y_scores
