"""Cluster quality metrics for PCA-based QC assessment."""
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from dataclasses import dataclass


@dataclass
class ClusterMetrics:
    silhouette: float           # -1 to 1, >0.5 = strong clustering
    calinski_harabasz: float    # Higher = better separation
    n_samples: int
    n_groups: int


def compute_cluster_metrics(
    scores: np.ndarray,
    labels: list[str],
) -> ClusterMetrics:
    """Compute clustering quality metrics on PCA scores.
    
    Args:
        scores: PCA scores array (n_samples, n_components)
        labels: Group labels for each sample
    """
    label_array = np.array(labels)
    
    sil = silhouette_score(scores, label_array)
    ch = calinski_harabasz_score(scores, label_array)
    
    return ClusterMetrics(
        silhouette=sil,
        calinski_harabasz=ch,
        n_samples=len(labels),
        n_groups=len(set(labels)),
    )
