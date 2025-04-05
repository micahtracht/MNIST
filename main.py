from kmeans import load_mnist
from least_squares import LeastSquaresClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # load dataset
    X, y = load_mnist(size=5000)
    X = X/255.0
    #X = (X-X.mean(axis=0))/X.std(axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # fit least-squares classifier for MNIST
    clf = LeastSquaresClassifier()
    clf.fit(X_train, y_train)

    # make preds & evaluate acc
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Least Squares accuracy:", acc)
