"""PVCA (Principal Variance Component Analysis) implementation."""
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class PVCAResult:
    """Container for PVCA results."""
    components: dict[str, float]  # factor -> proportion of variance
    n_pcs_used: int
    cumulative_variance_threshold: float


def run_pvca(
    df: pl.DataFrame,
    intensity_cols: list[str],
    conditions: list[str],
    threshold: float = 0.6,
) -> PVCAResult:
    """Run PVCA: PCA then variance decomposition per PC.
    
    Algorithm:
    1. Run PCA on transposed intensity matrix (samples as rows)
    2. Retain PCs explaining >= threshold cumulative variance
    3. For each retained PC, fit a linear model with condition + replicate
    4. Weight variance components by eigenvalues
    
    Args:
        df: DataFrame with standardized intensity columns
        intensity_cols: Column names for intensity values
        conditions: List of condition labels
        threshold: Cumulative variance threshold for PC retention
    
    Returns:
        PVCAResult with variance proportions per factor
    """
    from scipy import stats
    
    # Transpose: samples as rows, proteins as columns
    mat = df.select(intensity_cols).to_numpy().T.astype(np.float64)
    
    # Drop features with NaN
    valid_mask = ~np.isnan(mat).any(axis=0)
    mat_clean = mat[:, valid_mask]
    
    # Standardize: PVCA conventionally operates on the correlation matrix
    # (i.e. z-scored features).  This is part of the PVCA algorithm definition
    # (Boedigheimer et al.) and is correct here — unlike PCA-for-visualisation
    # where z-scoring after log-transform is NOT recommended.
    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat_clean)
    
    # Run PCA with all components
    n_comp = min(mat_scaled.shape)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(mat_scaled)
    explained = pca.explained_variance_ratio_
    
    # Determine how many PCs to retain
    cumsum = np.cumsum(explained)
    n_pcs = int(np.searchsorted(cumsum, threshold) + 1)
    n_pcs = min(n_pcs, len(explained))
    
    # Build factor vectors
    n_samples = len(intensity_cols)
    n_reps = n_samples // len(conditions)
    
    condition_labels = []
    replicate_labels = []
    for i, col in enumerate(intensity_cols):
        cond_idx = i // n_reps
        rep_idx = i % n_reps
        condition_labels.append(conditions[cond_idx])
        replicate_labels.append(rep_idx)
    
    # Encode factors numerically
    unique_conds = list(set(condition_labels))
    cond_encoded = np.array([unique_conds.index(c) for c in condition_labels], dtype=float)
    rep_encoded = np.array(replicate_labels, dtype=float)
    
    # For each PC, compute variance explained by each factor
    var_condition = 0.0
    var_replicate = 0.0
    var_residual = 0.0
    
    for pc_i in range(n_pcs):
        pc_scores = scores[:, pc_i]
        weight = explained[pc_i]
        
        # Total variance of this PC's scores
        total_var = np.var(pc_scores, ddof=0)
        if total_var == 0:
            continue
        
        # Compute SS for condition factor
        grand_mean = np.mean(pc_scores)
        ss_total = np.sum((pc_scores - grand_mean) ** 2)
        
        # SS condition
        ss_condition = 0
        for cond in unique_conds:
            mask = np.array([c == cond for c in condition_labels])
            group_mean = np.mean(pc_scores[mask])
            ss_condition += np.sum(mask) * (group_mean - grand_mean) ** 2
        
        # SS replicate (nested within condition)
        ss_replicate = 0
        for cond in unique_conds:
            cond_mask = np.array([c == cond for c in condition_labels])
            cond_scores = pc_scores[cond_mask]
            cond_reps = np.array(replicate_labels)[cond_mask]
            cond_mean = np.mean(cond_scores)
            for rep in set(cond_reps.tolist()):
                rep_mask = cond_reps == rep
                if np.sum(rep_mask) > 0:
                    rep_mean = np.mean(cond_scores[rep_mask])
                    ss_replicate += np.sum(rep_mask) * (rep_mean - cond_mean) ** 2
        
        ss_residual = max(0, ss_total - ss_condition - ss_replicate)
        
        # Weight by eigenvalue proportion
        var_condition += (ss_condition / ss_total) * weight if ss_total > 0 else 0
        var_replicate += (ss_replicate / ss_total) * weight if ss_total > 0 else 0
        var_residual += (ss_residual / ss_total) * weight if ss_total > 0 else 0
    
    # Normalize to sum to 1
    total = var_condition + var_replicate + var_residual
    if total > 0:
        var_condition /= total
        var_replicate /= total
        var_residual /= total
    
    return PVCAResult(
        components={
            "Condition": var_condition,
            "Replicate": var_replicate,
            "Residual": var_residual,
        },
        n_pcs_used=n_pcs,
        cumulative_variance_threshold=threshold,
    )
