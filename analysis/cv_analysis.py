"""CV analysis per intensity bin and per species."""
import numpy as np
import polars as pl


def compute_cvs(
    df: pl.DataFrame,
    intensity_cols: list[str],
    conditions: list[str] | None = None,
) -> pl.DataFrame:
    """Compute per-protein CV across replicates within each condition.
    
    Returns DataFrame with columns: protein_id, species, intensity_bin,
    cv_A, cv_B (one per condition).
    """
    result_cols = ["protein_id"]
    if "species" in df.columns:
        result_cols.append("species")
    if "intensity_bin" in df.columns:
        result_cols.append("intensity_bin")
    
    if conditions is None:
        conditions = ["A", "B"]
    
    # For each condition, compute CV = sd / mean * 100
    for cond in conditions:
        cond_cols = [c for c in intensity_cols if c.startswith(f"{cond}_")]
        if len(cond_cols) < 2:
            continue
        # Compute using numpy for reliability
        mat = df.select(cond_cols).to_numpy().astype(np.float64)
        means = np.nanmean(mat, axis=1)
        stds = np.nanstd(mat, axis=1, ddof=1)
        cvs = np.where(means > 0, (stds / means) * 100, np.nan)
        df = df.with_columns(pl.Series(f"cv_{cond}", cvs))
        result_cols.append(f"cv_{cond}")
    
    return df.select(result_cols)
