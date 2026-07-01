import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from ml_engine import SoftmaxRegressionFromScratch

def load_and_preprocess_data() -> tuple[np.ndarray, np.ndarray]:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "iris.csv")

    if not os.path.exists(DATA_PATH):
        url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
        df = pd.read_csv(url)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    class_mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
    df['variety'] = df['variety'].map(class_mapping)
    df = df.dropna()

    feature_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    X = df[feature_columns].values
    y = df['variety'].values.reshape(-1, 1)
    return X, y

def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0 

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    model = SoftmaxRegressionFromScratch(learning_rate=0.1, iterations=1000, lambda_param=0.1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test) * 100

    target_names = ['Setosa', 'Versicolor', 'Virginica']

    print("\n     MULTI-CLASS ENGINE EVALUATION REPORT         ")
    print(f"\nOverall Model Test Accuracy: {accuracy:.2f}%\n")
    print("Classification Metrics Performance:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("\nRaw Confusion Matrix Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
