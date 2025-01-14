import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def knn_search_algorithm(X, C, thresh_k):
    """
    KNN Search Algorithm for Cluster Dataset.

    Parameters:
        X (np.ndarray): Dataset points as a 2D array.
        C (list): Cluster labels corresponding to each point in X.
        thresh_k (int): Threshold for the maximum number of neighbors to consider.

    Returns:
        list: Predicted class for each data point in X.
    """
    m = len(C)
    k = 3  # Starting value of K

    # Initialize Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=thresh_k, metric='euclidean')
    nbrs.fit(X)

    predictions = []

    for x_i in X:
        # Find distances and indices of neighbors
        distances, indices = nbrs.kneighbors([x_i])
        
        # Iterate over K from 3 to thresh_k
        for k_val in range(k, thresh_k + 1):
            # Select K nearest neighbors and their labels
            k_indices = indices[0][:k_val]
            k_labels = [C[idx] for idx in k_indices]

            # Count occurrences of each class among neighbors
            class_count = Counter(k_labels)

            # Determine the majority class
            majority_class = max(class_count, key=class_count.get)

            # Check the stopping condition based on distances
            if distances[0][k_val - 1] >= distances[0][k - 1]:
                break

        predictions.append(majority_class)

    return predictions

# Example Usage
if __name__ == "__main__":
    # Example Dataset
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 points in 2D space
    C = np.random.choice(["Cluster_A", "Cluster_B", "Cluster_C"], size=100)  # Random cluster labels

    thresh_k = 10
    predicted_classes = knn_search_algorithm(X, C, thresh_k)

    # Display the results
    for i, pred_class in enumerate(predicted_classes):
        print(f"Point {i}: Predicted Class = {pred_class}")
