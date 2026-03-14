"""PCA computation engine using sklearn, operating on Polars DataFrames."""
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class PCAResult:
    """Container for PCA results."""
    scores: np.ndarray          # (n_samples, n_components)
    loadings: np.ndarray        # (n_features, n_components)
    explained_variance: np.ndarray  # Per-component explained variance ratio
    feature_names: list[str]
    sample_names: list[str]
    n_components: int


def run_pca(
    df: pl.DataFrame,
    intensity_cols: list[str],
    n_components: int = 5,
    scale: bool = False,
) -> PCAResult:
    """Run PCA on protein intensity matrix.

    Args:
        df: Polars DataFrame with protein rows and sample columns
        intensity_cols: Column names containing intensity values
        n_components: Number of PCs to retain
        scale: Whether to z-score standardize each protein before PCA.
               Default False — after log2/glog transformation, variance is
               already stabilised and z-scoring would amplify noise from
               low-abundance proteins. Set True only for raw/untransformed
               data or when unit-variance scaling is explicitly desired
               (e.g. when mixing very different measurement types).

    Returns:
        PCAResult with scores, loadings, and variance explained
    """
    # Transpose: PCA expects samples as rows
    mat = df.select(intensity_cols).to_numpy().T  # (n_samples, n_proteins)

    # Handle missing values: drop proteins with any NaN
    valid_mask = ~np.isnan(mat).any(axis=0)
    mat_clean = mat[:, valid_mask]

    # Get feature names for valid proteins
    if "protein_id" in df.columns:
        all_ids = df["protein_id"].to_list()
        feature_names = [all_ids[i] for i in np.where(valid_mask)[0]]
    else:
        feature_names = [f"feature_{i}" for i in np.where(valid_mask)[0]]

    if scale:
        scaler = StandardScaler()
        mat_clean = scaler.fit_transform(mat_clean)

    n_components = min(n_components, min(mat_clean.shape))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(mat_clean)

    return PCAResult(
        scores=scores,
        loadings=pca.components_.T,
        explained_variance=pca.explained_variance_ratio_,
        feature_names=feature_names,
        sample_names=intensity_cols,
        n_components=n_components,
    )
