import numpy as np

class VectorizedRegressionEngine:
    """A vectorized Linear Regression engine built from scratch."""
    def __init__(self, alpha: float = 0.01, iterations: int = 1000, tolerance: float = 1e-6):
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance  
        self.w = None 
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def _scale_features(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)
            self.sigma[self.sigma == 0] = 1.0 
        return (X - self.mu) / self.sigma

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        return np.sum((predictions - y) ** 2) / (2 * m)

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        m = X.shape[0]
        predictions = np.dot(X, self.w) + self.b
        errors = predictions - y
        dj_dw = np.dot(X.T, errors) / m
        dj_db = np.sum(errors) / m
        return dj_dw, dj_db

    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> list[tuple[int, float]]: 
        X_mat = np.array(X_raw, dtype=np.float64)
        y_vec = np.array(y, dtype=np.float64)
        X_scaled = self._scale_features(X_mat, training=True)
        self.w = np.zeros(X_scaled.shape[1])
        self.b = 0.0
        cost_history = []
        previous_cost = float('inf')
        
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X_scaled, y_vec)
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            if i % 10 == 0 or i == self.iterations - 1:
                current_cost = self.compute_cost(X_scaled, y_vec)
                cost_history.append((i, current_cost))
                if abs(previous_cost - current_cost) < self.tolerance:
                    break
                previous_cost = current_cost
        return cost_history

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        if self.w is None or self.mu is None:
            raise ValueError("Model not fitted yet.")
        X_mat = np.array(X_raw, dtype=np.float64)
        X_scaled = self._scale_features(X_mat, training=False)
        return np.dot(X_scaled, self.w) + self.b


class SoftmaxRegressionFromScratch:
    """A vectorized Multi-Class Softmax Regression engine built from scratch."""
    def __init__(self, learning_rate: float = 0.05, iterations: int = 2000, lambda_param: float = 0.1):
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        self.__lambda_param = lambda_param
        self.__w = None  
        self.__b = None  
        self.__cost_history = []

    def __softmax(self, z: np.ndarray) -> np.ndarray:
        exp_shifted = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def __compute_cost(self, y_one_hot: np.ndarray, y_hat: np.ndarray) -> float:
        m = y_one_hot.shape[0]
        epsilon = 1e-15
        y_hat_stable = np.clip(y_hat, epsilon, 1 - epsilon)
        base_cost = - (1 / m) * np.sum(y_one_hot * np.log(y_hat_stable))
        reg_penalty = (self.__lambda_param / (2 * m)) * np.sum(np.square(self.__w))
        return float(base_cost + reg_penalty)

    def __compute_gradients(self, X: np.ndarray, y_one_hot: np.ndarray, y_hat: np.ndarray) -> tuple:
        m = X.shape[0]
        error = y_hat - y_one_hot
        dw = (1 / m) * np.dot(X.T, error) + (self.__lambda_param / m) * self.__w
        db = (1 / m) * np.sum(error, axis=0, keepdims=True)
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        m, n = X.shape
        num_classes = len(np.unique(y))
        y_one_hot = np.zeros((m, num_classes))
        y_one_hot[np.arange(m), y.flatten()] = 1 
        self.__w = np.zeros((n, num_classes))
        self.__b = np.zeros((1, num_classes))
        self.__cost_history = []

        for i in range(self.__iterations):
            z = np.dot(X, self.__w) + self.__b
            y_hat = self.__softmax(z)
            cost = self.__compute_cost(y_one_hot, y_hat)
            self.__cost_history.append(cost)
            dw, db = self.__compute_gradients(X, y_one_hot, y_hat)
            self.__w -= self.__learning_rate * dw
            self.__b -= self.__learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.__w is None or self.__b is None:
            raise ValueError("Model not fitted yet.")
        z = np.dot(X, self.__w) + self.__b
        return self.__softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat = self.predict_proba(X)
        return np.argmax(y_hat, axis=1).reshape(-1, 1)

    @property
    def w(self): return self.__w

    @property
    def b(self): return self.__b

    @property
    def cost_history(self): return self.__cost_history
