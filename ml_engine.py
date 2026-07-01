import numpy as np

class SoftmaxRegressionFromScratch:
    def __init__(self, learning_rate: float = 0.05, iterations: int = 2000, lambda_param: float = 0.1):
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        self.__lambda_param = lambda_param
        self.__w = None  # Shape will be: (num_features, num_classes)
        self.__b = None  # Shape will be: (1, num_classes)
        self.__cost_history = []

    def __softmax(self, z: np.ndarray) -> np.ndarray:
        """
        BLOCK 1: Multi-Class Activation Engine with Exponential Overflow Protection.
        Subtracting the maximum row value keeps the exponent scores <= 0.
        """
        # Max subtraction trick prevents e^z from crashing into np.inf
        exp_shifted = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def __compute_cost(self, y_one_hot: np.ndarray, y_hat: np.ndarray) -> float:
        m = y_one_hot.shape[0]
        epsilon = 1e-15
        y_hat_stable = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Categorical cross-entropy formula
        base_cost = - (1 / m) * np.sum(y_one_hot * np.log(y_hat_stable))
        
        # L2 Regularization penalty across the entire weight matrix
        reg_penalty = (self.__lambda_param / (2 * m)) * np.sum(np.square(self.__w))
        
        return float(base_cost + reg_penalty)

    def __compute_gradients(self, X: np.ndarray, y_one_hot: np.ndarray, y_hat: np.ndarray) -> tuple:
        m = X.shape[0]
        error = y_hat - y_one_hot  # Shape: (m, num_classes)
        
        # Multi-variable matrix derivative calculation
        dw = (1 / m) * np.dot(X.T, error) + (self.__lambda_param / m) * self.__w
        db = (1 / m) * np.sum(error, axis=0, keepdims=True)
        
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """BLOCK 4: Multi-Class Optimization Training Loop."""
        m, n = X.shape
        num_classes = len(np.unique(y))

        y_one_hot = np.zeros((m, num_classes))
        y_one_hot[np.arange(m), y.flatten()] = 1 
        
        self.__w = np.zeros((n, num_classes))
        self.__b = np.zeros((1, num_classes))
        self.__cost_history = []

        for i in range(self.__iterations):
            # Forward Pass
            z = np.dot(X, self.__w) + self.__b
            y_hat = self.__softmax(z)

            # Evaluate Cost
            cost = self.__compute_cost(y_one_hot, y_hat)
            self.__cost_history.append(cost)

            # Backward Pass
            dw, db = self.__compute_gradients(X, y_one_hot, y_hat)

            # Gradient Descent Coordinate State Mutation
            self.__w -= self.__learning_rate * dw
            self.__b -= self.__learning_rate * db

            if i % (self.__iterations // 10) == 0 or i == self.__iterations - 1:
                print(f"Iteration {i:5d} | Categorical Cross-Entropy Cost: {cost:.6f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """BLOCK 5: Probabilistic Inference (Returns likelihood distribution per class)"""
        if self.__w is None or self.__b is None:
            raise ValueError("Model state uninitialized. Run fit() first.")
        z = np.dot(X, self.__w) + self.__b
        return self.__softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """BLOCK 5: Class Label Inference (Returns index of maximum probability)"""
        y_hat = self.predict_proba(X)
        return np.argmax(y_hat, axis=1).reshape(-1, 1)

    @property
    def w(self): return self.__w

    @property
    def b(self): return self.__b

    @property
    def cost_history(self): return self.__cost_history