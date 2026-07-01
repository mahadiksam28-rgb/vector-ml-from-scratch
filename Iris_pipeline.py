import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import SoftmaxRegressionFromScratch


print("--- Requesting Real-World Iris Dataset ---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "iris.csv")

if not os.path.exists(DATA_PATH):
    print("Local CSV file not detected. Pulling directly from server repository...")
    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# ==========================================================
# STEP 2: DATA CLEANING, ENCODING & RE-STRUCTURING
# ==========================================================
class_mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
df['variety'] = df['variety'].map(class_mapping)
df = df.dropna()

feature_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
X = df[feature_columns].values
y = df['variety'].values.reshape(-1, 1)

print(f"Data Cleaning Completed. Feature Matrix Shape: {X.shape}")

# ==========================================================
# STEP 3: STRATIFIED TRAIN-TEST ISOLATION SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================================
# STEP 4: HIGH-DIMENSIONAL FEATURE SCALING (Z-SCORE)
# ==========================================================
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# ==========================================================
# STEP 5: MODEL INITIALIZATION & TRAINING WORKSPACE
# ==========================================================
print("\n--- Deploying Custom Softmax Regression Model ---")
model = SoftmaxRegressionFromScratch(learning_rate=0.1, iterations=1000, lambda_param=0.1)

print("Running Multi-Class Gradient Descent...")
model.fit(X_train_scaled, y_train)

# ==========================================================
# STEP 6: ADVANCED TEST PERFORMANCE EVALUATION
# ==========================================================
y_pred = model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test) * 100
print(f"\n>>> Multi-Class Out-of-Sample Accuracy: {accuracy:.2f}% <<<")

# Professional Industry Diagnostics Layer
print("\n--- Detailed Production Classification Report ---")
target_names = ['Setosa', 'Versicolor', 'Virginica']
print(classification_report(y_test, y_pred, target_names=target_names))

print("Raw Confusion Matrix Metrics:")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# STEP 7: PRODUCTION RECOGNITION ANALYSIS VISUALIZATION
# ==========================================================
plt.figure(figsize=(7, 5))
plt.plot(model.cost_history, color='teal', linewidth=2.5, label='Categorical Cross-Entropy Loss')
plt.title('Multi-Class Optimization Tracking Curve', fontsize=12, fontweight='bold')
plt.xlabel('Gradient Descent Updates (Iterations)', fontsize=10)
plt.ylabel('Calculated Loss Magnitudes', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()