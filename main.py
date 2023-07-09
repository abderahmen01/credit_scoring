import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
import time

from rotation_forest_class import RotationForestClassifier

if __name__ == "__main__":
    # Load the training data
    data = pd.read_csv('data3.csv')

    # Extract the features (X) and labels (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Instantiate the RotationForestClassifier
    rotation_forest = RotationForestClassifier(k=5, n_features=25, n_estimators=10)

    from sklearn.model_selection import cross_validate

    # Define scoring metrics
    # Define scoring metrics for binary classification
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'time': make_scorer(time.time)
    }

    # Perform 10-fold cross-validation
    scores = cross_validate(rotation_forest, X, y, scoring=scoring_metrics, cv=10)

    # Compute average scores for each metric
    average_scores = {metric: np.mean(scores['test_' + metric]) for metric in scoring_metrics.keys()}
    scores = cross_validate(rotation_forest, X, y, scoring=scoring_metrics, cv=5)



    # Print average scores
    for metric, score in average_scores.items():
        print(f"{metric}: {score}")













