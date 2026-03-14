"""Intraclass Correlation Coefficients for replicate assessment."""
import numpy as np
import polars as pl
import pingouin as pg
import pandas as pd


def compute_icc(
    df: pl.DataFrame,
    intensity_cols: list[str],
    conditions: list[str] | None = None,
) -> dict[str, float]:
    """Compute ICC(3,1) — two-way mixed, single measures — per condition.
    
    This measures consistency of replicate measurements.
    ICC > 0.9 = excellent, 0.75-0.9 = good, 0.5-0.75 = moderate, <0.5 = poor
    
    Args:
        df: DataFrame with standardized columns
        intensity_cols: All intensity column names
        conditions: List of condition labels
    
    Returns:
        Dict mapping condition -> ICC value
    """
    if conditions is None:
        conditions = ["A", "B"]
    
    results = {}
    
    for cond in conditions:
        cond_cols = [c for c in intensity_cols if c.startswith(f"{cond}_")]
        if len(cond_cols) < 2:
            continue
        
        # Build long-format data for pingouin
        mat = df.select(cond_cols).to_numpy().astype(np.float64)
        
        # Remove rows with any NaN
        valid_rows = ~np.isnan(mat).any(axis=1)
        mat_valid = mat[valid_rows]
        
        if len(mat_valid) < 3:
            results[cond] = np.nan
            continue
        
        # Build long format: targets (proteins), raters (replicates), ratings
        n_proteins = mat_valid.shape[0]
        n_reps = mat_valid.shape[1]
        
        targets = np.repeat(np.arange(n_proteins), n_reps)
        raters = np.tile(np.arange(n_reps), n_proteins)
        ratings = mat_valid.flatten()
        
        long_df = pd.DataFrame({
            "targets": targets,
            "raters": raters,
            "ratings": ratings,
        })
        
        try:
            icc_result = pg.intraclass_corr(
                data=long_df,
                targets="targets",
                raters="raters",
                ratings="ratings",
            )
            # ICC3 = two-way mixed, single measures (consistency)
            icc3_row = icc_result[icc_result["Type"] == "ICC3"]
            if len(icc3_row) > 0:
                results[cond] = float(icc3_row["ICC"].values[0])
            else:
                results[cond] = float(icc_result.iloc[2]["ICC"])
        except Exception:
            results[cond] = np.nan
    
    return results
