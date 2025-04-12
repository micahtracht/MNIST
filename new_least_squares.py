import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import time

def load_data():
    print('Data loading')
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    print("Data loaded")
    return mnist

# Load and preprocess data
mnist = load_data()
X = mnist.data.astype('float64')
y = mnist.target.astype('int64')
X /= 255  # normalize X

# Add bias term
X_bias = np.c_[np.ones((X.shape[0], 1)), X]
print(f"Original features shape: {X.shape}")
print(f"Features shape with bias: {X_bias.shape}")

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False, categories='auto')
Y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
print(f"Labels shape (one-hot encoded): {Y_onehot.shape}")
print(f"Example label: {y.iloc[0]}, One-hot: {Y_onehot[0]}")

# Splitting data with one-hot labels for training
print('splitting data')
X_train, X_test, Y_train_onehot, Y_test_onehot = train_test_split(
    X_bias, Y_onehot, test_size=0.2, random_state=42, stratify=y
)

print('---diagnostic data---')
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape (one-hot): {Y_train_onehot.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape (one-hot): {Y_test_onehot.shape}")

# Calculate weights using least squares
print("Calculating weights using least squares (np.linalg.lstsq)...")
start_time = time.time()
W, residuals, rank, s = np.linalg.lstsq(X_train, Y_train_onehot, rcond=None)
end_time = time.time()
print(f"Weight calculation took {end_time - start_time:.2f} seconds.")
print(f"Weight matrix W shape: {W.shape}")  # Should be (785, 10)

# Predicting on test set
Y_pred_raw = X_test @ W
y_pred = np.argmax(Y_pred_raw, axis=1)

# Converting one-hot encoded test labels back to original labels for evaluation
y_test_labels = np.argmax(Y_test_onehot, axis=1)

print("Evaluating model performance...")
accuracy = accuracy_score(y_test_labels, y_pred)
report = classification_report(y_test_labels, y_pred)

print(f"\nAccuracy on Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)
