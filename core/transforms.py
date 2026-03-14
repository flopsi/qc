"""Transformation functions: log2, glog (Python-native), VSN (via rpy2)."""
import numpy as np
import polars as pl


def log2_transform(
    df: pl.DataFrame,
    intensity_cols: list[str],
    pseudocount: float = 1.0,
) -> pl.DataFrame:
    """Apply log2(x + pseudocount) to intensity columns."""
    return df.with_columns([
        (pl.col(c) + pseudocount).log(base=2).alias(c)
        for c in intensity_cols
    ])


def glog_transform(
    df: pl.DataFrame,
    intensity_cols: list[str],
    lam: float | None = None,
) -> pl.DataFrame:
    """Generalized log transform: glog(x) = log2(x + sqrt(x^2 + lambda)).
    
    If lambda is None, estimate it from the data using the method of
    Rocke & Durbin (2003). The glog approaches log2 at high intensities
    and compresses variance at low intensities, similar to VSN.
    """
    # Extract numpy array for computation
    mat = df.select(intensity_cols).to_numpy().astype(np.float64)
    
    if lam is None:
        # Estimate lambda: use median of row-wise variances as starting point
        row_vars = np.nanvar(mat, axis=1)
        lam = float(np.nanmedian(row_vars))
    
    # Apply glog: log2(x + sqrt(x^2 + lambda))
    transformed = np.log2(mat + np.sqrt(mat**2 + lam))
    
    # Rebuild DataFrame
    transformed_df = pl.DataFrame(
        {col: transformed[:, i] for i, col in enumerate(intensity_cols)}
    )
    non_intensity = [c for c in df.columns if c not in intensity_cols]
    return pl.concat([df.select(non_intensity), transformed_df], how="horizontal")


def vsn_transform_rpy2(
    df: pl.DataFrame,
    intensity_cols: list[str],
) -> pl.DataFrame:
    """Apply VSN via R's vsn package through rpy2.
    Falls back to glog_transform if rpy2/vsn not available.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        
        vsn_pkg = importr("vsn")
        mat = df.select(intensity_cols).to_numpy().astype(np.float64)
        
        # VSN expects features as rows, samples as columns
        r_mat = ro.r.matrix(ro.FloatVector(mat.T.flatten()),
                            nrow=mat.shape[1], ncol=mat.shape[0])
        
        # Run vsn2 and extract normalized data
        fit = vsn_pkg.vsn2(r_mat)
        normalized = np.array(ro.r.exprs(fit)).T
        numpy2ri.deactivate()
        
        transformed_df = pl.DataFrame(
            {col: normalized[:, i] for i, col in enumerate(intensity_cols)}
        )
        non_intensity = [c for c in df.columns if c not in intensity_cols]
        return pl.concat([df.select(non_intensity), transformed_df], how="horizontal")
    except (ImportError, Exception) as e:
        import warnings
        warnings.warn(
            f"VSN via rpy2 failed ({e}). Falling back to glog transform."
        )
        return glog_transform(df, intensity_cols)


def compute_intensity_bins(
    df: pl.DataFrame,
    intensity_cols: list[str],
    n_bins: int = 4,
) -> pl.DataFrame:
    """Add 'intensity_bin' column (Q1-Q4) based on mean log2 intensity across samples."""
    # Compute row-wise mean of intensity columns
    mat = df.select(intensity_cols).to_numpy().astype(np.float64)
    row_means = np.nanmean(mat, axis=1)
    
    # Compute quantile boundaries
    valid_means = row_means[~np.isnan(row_means)]
    boundaries = [np.percentile(valid_means, 100 * i / n_bins) for i in range(1, n_bins)]
    
    # Assign bins
    bins = np.full(len(row_means), f"Q{n_bins}", dtype=object)
    for i in range(len(boundaries) - 1, -1, -1):
        bins[row_means <= boundaries[i]] = f"Q{i + 1}"
    bins[np.isnan(row_means)] = "NA"
    
    return df.with_columns(pl.Series("intensity_bin", bins.tolist()))
