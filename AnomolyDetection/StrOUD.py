from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator
import numpy as np


class StrOUD(BaseEstimator):

    def __init__(self, k_neighbors=3, metric='euclidean', confidence=0.95):

        self.k_neighbors = k_neighbors
        self.metric = metric
        self.confidence = confidence

        self.X = None
        self.y = None

        self.lof_scores = None
        self.p_vals = None
        self.pred = None

        return

    def fit(self, X, y=None):
        self.X = X
        self.y = y

        return

    def transform(self, data):

        self.lof_scores = np.array([
            compute_lof_score(vector, data=self.X, metric=self.metric, k_neighbors=self.k_neighbors)
            for vector in data
        ])

        self.p_vals = [
            1 - len(self.lof_scores[np.where(self.lof_scores <= score)]) / len(self.lof_scores)
            for score in self.lof_scores
        ]

        return

    def predict(self, data):

        if self.p_vals is None:
            self.transform(data)

        self.pred = [1 if p_val >= self.confidence else 0 for p_val in self.p_vals]

        return self.pred


def compute_dataset_distances(test_vector, data, metric):

    if metric == 'euclidean':

        distances = []

        for dataset_vector in data:
            if np.array_equal(test_vector, dataset_vector):
                continue
            distances.append((euclidean(test_vector, dataset_vector), dataset_vector))

    else:
        raise ValueError('At the moment, only Euclidean metric is supported')

    return distances


def compute_vector_knn_distances(distances, k_neighbors):
    distances.sort()

    return distances[:k_neighbors]


def compute_lof_score(vector, data, metric, k_neighbors):

    vector_distances = compute_dataset_distances(test_vector=vector, data=data, metric=metric)
    vector_knn = compute_vector_knn_distances(distances=vector_distances, k_neighbors=k_neighbors)
    knn_vector_distances = [knn[0] for knn in vector_knn]
    knn_vectors = [knn[1] for knn in vector_knn]

    vector_reachability = max(knn_vector_distances)

    neighbor_distances = [
        [knn[0] for knn in compute_dataset_distances(test_vector=neighbor_vector, data=data, metric=metric)]
        for neighbor_vector in knn_vectors
    ]

    neighbors_knn = [
        compute_vector_knn_distances(distances=distances, k_neighbors=k_neighbors)
        for distances in neighbor_distances
    ]

    neighbors_reachability = [max(k_distances) for k_distances in neighbors_knn]
    average_neighbor_reachability = np.average(neighbors_reachability)

    return average_neighbor_reachability / vector_reachability
