import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
SEED = 42

def k_means(vectors, k, max_iter=1000, tol=0.01):
    """
    Performs k-means clustering on a list of n-dimensional vectors.
    
    Parameters:
        vectors (list): List of vectors
        k (int): Number of clusters.
        max_iter (int): Max number of iterations.
        tol (float): Tolerance for convergence (algorithm stops when the change in cost is below tol).
    
    Returns:
        cost (float): Final cost (avg squared distance).
        assignments (list): List of [vector, cluster_index] pairs.
        centroids (list): List of k centroid vectors.
        costs (list): The cost (avg squared distance) at each iteration.
        history (list): (assignments, centroids) at each iteration
    """
    # Helper to compute squared Euclidean distance
    def dist_squared(u, v):
        return sum((u[i] - v[i])**2 for i in range(len(u)))
    
    # Compute cost as the avg squared distance of points to their assigned centroid (euclidean)
    def j_clust(assignments, centroids):
        total = 0.0
        for vec, group in assignments:
            total += dist_squared(vec, centroids[group])
        return total / len(assignments)
    
    # Assign each vector to nearest centroid
    def assign_groups(assignments, centroids):
        for i, (vec, _) in enumerate(assignments):
            best_group = 0
            min_dist = float('inf')
            for j in range(k):
                d = dist_squared(vec, centroids[j])
                if d < min_dist:
                    min_dist = d
                    best_group = j
            assignments[i][1] = best_group
    
    # Update centroids to be the mean of all vectors assigned to each cluster
    def assign_centroids(assignments, centroids):
        groups = {i: [] for i in range(k)}
        for vec, group in assignments:
            groups[group].append(vec)
        for j in range(k):
            if groups[j]:  # if the cluster is not empty, compute its mean
                new_centroid = [sum(coords)/len(coords) for coords in zip(*groups[j])]
                centroids[j] = new_centroid
            else:
                # If no vectors are assigned to this cluster, reinitialize its centroid randomly.
                centroids[j] = list(np.random.rand(len(vectors[0])))
    
    # Initialize assignments (each vector is paired with 0 as a default cluster)
    assignments = [[vec, 0] for vec in vectors]
    
    # Initialize centroids with random vectors
    centroids = [list(np.random.rand(len(vectors[0]))) for _ in range(k)]
    
    prev_cost = float('inf')
    cost = j_clust(assignments, centroids)
    num_iter = 0
    
    # Iterate until convergence or maximum iterations reached.
    costs = []
    history = []
    while abs(prev_cost - cost) > tol and num_iter < max_iter:
        num_iter += 1
        assign_groups(assignments, centroids)
        assign_centroids(assignments, centroids)
        prev_cost = cost
        cost = j_clust(assignments, centroids)
        costs.append(cost)
        
        assignment_snapshot = [list(a) for a in assignments]
        centroid_snapshot = [list(c) for c in centroids]
        history.append((assignment_snapshot, centroid_snapshot))
    
    return cost, assignments, centroids, costs, history

def cluster_assigner(vector, centroids): # takes unseen vector and returns it's nearest centroid
    
    #Helper function to compute squared distance
    def dist_squared(u, v):
        return sum((u[i] - v[i])**2 for i in range(len(u)))

    min_dist = float('inf')
    min_idx = -1
    for i, c in enumerate(centroids):
        squared_distance = dist_squared(vector, c)
        if squared_distance < min_dist:
            min_dist = squared_distance
            min_idx = i
    return min_idx #  returns minimum index, aligns with cluster map

def validate_new(vector, centroids, cluster_digit_map, true_value):
    cluster_idx = cluster_assigner(vector, centroids)
    pred_digit = cluster_digit_map.get(cluster_idx, -1)
    return pred_digit == true_value

def load_mnist(size=4000):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(int)

    idx = np.random.choice(len(X), size, replace=False)
    return X[idx], y[idx]

def generate_maps(k):
    cluster_labels = [[] for i in range(k)]
    for (vec, cluster), label in zip(assignments, y_sample):
        cluster_labels[cluster].append(label)

    cluster_digit_map = {}
    for i in range(k):
        if cluster_labels[i]:
            cluster_digit_map[i] = np.argmax(np.bincount(np.array(cluster_labels[i])))
        else:
            cluster_digit_map[i] = -1

    y_pred = [cluster_digit_map[cluster] for _, cluster in assignments]
    return cluster_digit_map, y_pred

def build_cluster_digit_map(assignments, labels, k):
    cluster_labels = [[] for _ in range(k)]
    for(_, cluster), label in zip(assignments, labels):
        cluster_labels[cluster].append(label)
        
    cluster_digit_map = {}
    for i in range(k):
        if cluster_labels[i]:
            cluster_digit_map[i] = np.argmax(np.bincount(np.array(cluster_labels[i])))
        else:
            cluster_digit_map[i] = -1
    return cluster_digit_map
def predict(X, centroids, cluster_digit_map):
    preds = []
    for vec in X:
        cluster = cluster_assigner(vec, centroids)
        digit = cluster_digit_map.get(cluster, -1)
        preds.append(digit)
    return preds
# evaluate the model on unseen data

X_sample, y_sample = load_mnist()
cost, assignments, centroids, costs, history = k_means(X_sample.tolist(), k=10)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.25, random_state=SEED)
cluster_digit_map = build_cluster_digit_map(assignments, y_train, k=10)

y_test_pred = predict(X_test, centroids, cluster_digit_map)
print("Accuracy:", accuracy_score(y_test, y_test_pred))




