# Vectorized ML Engines From Scratch

A production-ready collection of foundational supervised machine learning engines implemented entirely from scratch using vectorized NumPy operations. This repository demonstrates the underlying mathematics of linear optimization models, avoiding reliance on high-level ML frameworks for core mathematical steps.

## 🛠️ Engine Architecture & Components

### 1. Vectorized Linear Regression Engine
* **Purpose:** Continuous numerical target estimation (e.g., Car Price Prediction).
* **Optimization:** Mean Squared Error (MSE) minimization via vectorized Gradient Descent.
* **Features:** Built-in automated Z-score feature scaling and dynamic early-stopping convergence criteria based on loss tolerances.

### 2. Multi-Class Softmax Regression Engine
* **Purpose:** Categorical classification across mutually exclusive target domains (e.g., Iris Flower Classification).
* **Optimization:** Categorical Cross-Entropy objective function optimized with $L_2$ Regularization weight-decay penalties.
* **Stability:** Integrated exponential numerical overflow protection (maximum row-shifting translation) within the activation layer.

---

## 📁 Repository Blueprint

```text
├── ml_engine.py                  # Core vectorized algorithmic classes
├── main_car_prediction.py        # Automated linear regression pipeline
├── main_iris_classification.py   # Automated multi-class classification pipeline
├── .gitignore                    # Local storage restriction rule-map
└── requirements.txt              # Production dependency environment maps
