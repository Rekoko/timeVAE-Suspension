# This file creates an evaluation on the performance of a VAE generator based on
# selected time series metrics.
import pycatch22
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

def compute_avg_wasserstein(real_array, fake_array):
    n_features = real_array.shape[1]

    feature_distances = []
    
    for i in range(n_features):
        real_feat = real_array[:, i]
        fake_feat = fake_array[:, i]

        mu = np.mean(real_feat)
        std = np.std(real_feat)
        if std == 0:
            std = 1e-8

        real_norm = (real_feat - mu) / std
        fake_norm = (fake_feat - mu) / std

        dist = wasserstein_distance(real_norm, fake_norm)
        feature_distances.append(dist)

    average_distance = np.mean(feature_distances)

    return {
        "feature_distances": feature_distances,
        "average_distance": average_distance
    }


def compute_catch_22_scores(real_array, fake_array):
    """
    Computes generation features from the catch22 library
    """
    real_scores = [pycatch22.catch22_all(ts) for ts in real_array]
    fake_scores = [pycatch22.catch22_all(ts) for ts in fake_array]

    indices_to_drop = [7, 8, 9, 12, 16, 17, 21] # Indices that are classification focused

    real_scores = np.delete(real_scores, indices_to_drop, axis=1)
    fake_scores = np.delete(fake_scores, indices_to_drop, axis=1)

    return real_scores, fake_scores