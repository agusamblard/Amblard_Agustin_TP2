import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegressionL2:
    def __init__(self, dataset=None, target_name='y', ridge_lambda=0, fit=True, use_class_weights=False):
        self.ridge_lambda = ridge_lambda
        self.use_class_weights = use_class_weights
        self.fitted = False
        self.ovr_models = []

        if dataset is not None:
            assert isinstance(dataset, pd.DataFrame)
            dataset_X = dataset.drop(columns=[target_name])
            X = np.array(dataset_X.values).astype(float)
            y = np.array(dataset[target_name].values)
            self.features = np.array(['intercept'] + list(dataset_X.columns))
            self._initialize_data(X, y)

            if fit:
                self.fit()

    def _initialize_data(self, X, y):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = y
        self.w = np.zeros(self.X.shape[1])
        self.w_trace = [self.w.copy()]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, w, use_class_weights=None):
        if use_class_weights is None:
            use_class_weights = self.use_class_weights

        n = len(self.y)
        predictions = self.sigmoid(self.X @ w)
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)

        if use_class_weights:
            class_counts = np.bincount(self.y.astype(int))
            pi_min = class_counts.min() / len(self.y)
            pi_max = class_counts.max() / len(self.y)
            C = pi_max / pi_min

            minority_class = class_counts.argmin()
            sample_weights = np.where(self.y == minority_class, C, 1)

            log_loss = -(1 / n) * np.sum(sample_weights * (
                self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions)
            ))
        else:
            log_loss = -(1 / n) * (
                self.y @ np.log(predictions) + (1 - self.y) @ np.log(1 - predictions)
            )

        # ⬇️ Agregamos regularización L2 al final
        loss = log_loss + self.ridge_lambda * np.sum(w ** 2)

        return loss

    def gradient(self, w):
        n = len(self.y)
        predictions = self.sigmoid(self.X @ w)
        gradient = (1 / n) * self.X.T @ (predictions - self.y)

        regularization_term = (self.ridge_lambda / n) * w
        regularization_term[0] = 0
        gradient += regularization_term

        return gradient

    def gradient_descent(self, x0, lr=0.01, tol=1e-10, max_iter=50000):
        xk = x0
        self.w_trace = [x0]
        k = 0
        loss_values = [self.loss(x0, use_class_weights=self.use_class_weights)]

        while k < max_iter:
            grad = self.gradient(xk)
            if np.linalg.norm(grad) < tol:
                break

            xk = xk - lr * grad
            self.w_trace.append(xk.copy())

            current_loss = self.loss(xk, use_class_weights=self.use_class_weights)
            loss_values.append(current_loss)

            if abs(loss_values[-1] - loss_values[-2]) < tol:
                break

            k += 1

        return xk

    def fit(self):
        unique_classes = np.unique(self.y)
        self.classes_ = unique_classes  # <-- Esta línea nueva

        if len(unique_classes) == 2:
            self.w = self.gradient_descent(self.w)
            self.fitted = True
        else:
            self.ovr_models = []
            for c in unique_classes:
                binary_y = (self.y == c).astype(int)

                model = LogisticRegressionL2(
                    ridge_lambda=self.ridge_lambda,
                    use_class_weights=self.use_class_weights,
                    fit=False
                )
                model._initialize_data(self.X[:, 1:], binary_y)
                model.fit()
                self.ovr_models.append(model)
            self.fitted = True


    def predict_proba(self, X_new):
        X_new = np.column_stack((np.ones(X_new.shape[0]), X_new))
        if hasattr(self, 'ovr_models') and len(self.ovr_models) > 0:
            probs = np.column_stack([model.sigmoid(X_new @ model.w) for model in self.ovr_models])
            return probs  # Las columnas ahora están en el orden de self.classes_
        else:
            z = X_new @ self.w
            z = np.clip(z, -100, 100)
            return self.sigmoid(z)

    def predict(self, X_new, threshold=0.5):
        proba = self.predict_proba(X_new)
        if len(proba.shape) == 1:
            return (proba >= threshold).astype(int)
        else:
            indices = np.argmax(proba, axis=1)
            return np.array([self.classes_[i] for i in indices])
