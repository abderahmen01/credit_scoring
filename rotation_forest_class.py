import numpy as np
import rsfs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, n_features=5, n_estimators=10):
        self.k = k
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.rsfs_params = {
            'RSFS': {
                'Classifier': 'KNN',
                'Classifier Properties': {
                    'n_neighbors': 3,
                    'weights': 'distance'
                },
                'Dummy feats': 25,
                'delta': 0.05,
                'maxiters': 100,
                'fn': 'sqrt',
                'cutoff': 0.99,
                'Threshold': 5,
            },
            'Verbose': 1
        }
        self.rf_ensemble = []

    def fit(self, X, y):
        global rotation_matrix
        self.rf_ensemble = []
        for a in range(self.n_estimators):
            rf = RandomForestClassifier(max_features=self.n_features)
            data_train, data_test, label_train, label_test = train_test_split(
                X, y, test_size=0.10, random_state=42, stratify=y)
            binary_vectors = []
            for i in range(self.k):
                results = rsfs.RSFS(data_train, data_test, label_train, label_test, self.rsfs_params)
                selected_features = results['F_RSFS']
                binary_vector = np.zeros(X.shape[1], dtype=int)
                binary_vector[selected_features] = 1
                binary_vectors.append(binary_vector)

                print(f"\tRSFS Subset {i + 1}/{self.k}")

            rotation_matrix = np.vstack(binary_vectors).T
            rotate_train = np.dot(X, rotation_matrix)
            print(rotate_train.shape)
            rf.fit(rotate_train, y)
            self.rf_ensemble.append(rf)
            print(f"estimator {a + 1}/{self.n_estimators}")

    def predict(self, X):
        ensemble_predictions = []
        for rf in self.rf_ensemble:
            rotate_X = np.dot(X, rotation_matrix)  # Apply rotation transformation to input data
            predictions = rf.predict(rotate_X)  # Make predictions on the rotated data
            print(predictions)
            ensemble_predictions.append(predictions)
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        binary_predictions = np.where(mean_predictions < 0.5, 0, 1)
        return binary_predictions


