import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from matrix import Matrix
import time

SEED = 42

def load_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    return mnist

mnist = load_data()
X = mnist.data.astype('float64') # fixes the data type to floats, ensuring no errors.
y = mnist.target.astype('int64') # fixes the label types to ints, rather than something custom (or strings), so our model can learn them with least squares.

X /= 255 # normalize

X = X[:500]
y = y[:500]

X_bias = np.c_[np.ones((X.shape[0], 1)), X] # adds a column of ones, so that each data set has a 1 it can use as an offset. This lets the model shift its y intercept. np.c concatenates horizontally, and np.ones makes a matrix of ones according to the specified shape (in this case, X.shape[0], or num rows tall, and 1 wide, so it's a column vector.)

encoder = OneHotEncoder(sparse_output=False, categories='auto')
Y_onehot = encoder.fit_transform(y.values.reshape(-1, 1)) # makes sure labels are a 2d array, as needed by onehotencoder.

X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(
    X_bias, Y_onehot, test_size=0.2, random_state=SEED, stratify=y
)


X_train_matrix = Matrix(X_train)
y_train_onehot_matrix = Matrix(y_train_onehot)
X_test_matrix = Matrix(X_test)
y_test_onehot_matrix = Matrix(y_test_onehot)

print('started calculations')
start_time = time.time()
W = X_train_matrix.pseudoinverse().matmul(y_train_onehot_matrix) # Finds x dagger y
end_time = time.time()

print(f"Weight calculation time: {end_time - start_time:.3f} seconds")
print(f"Weight matrix (W) shape:", W.shape) # should be (785, 10)

y_pred_raw = X_test_matrix.matmul(W)
y_pred = np.argmax(y_pred_raw, axis=1) # y_pred_raw is an array of assigned probabilities, but we can only guess one number, so we pick the highest one (axis=1) using argmax.
y_test_labels = np.argmax(y_test_onehot, axis=1)

acc = accuracy_score(y_test_labels, y_pred)
report = classification_report(y_test_labels, y_pred)
print(acc)
print(report)