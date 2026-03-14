"""Mean-SD diagnostic plot for assessing variance stabilization."""
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d


def create_mean_sd_plot(
    intensity_matrix: np.ndarray,
    title: str = "Mean-SD Plot",
    n_bins: int = 50,
) -> go.Figure:
    """Create Mean-SD plot: row-wise mean vs. row-wise SD.
    
    A good variance-stabilizing transformation should show a flat
    trend (SD independent of mean).
    
    Args:
        intensity_matrix: (n_proteins, n_samples) array (already transformed)
        title: Plot title
        n_bins: Number of bins for the loess-like smoother
    """
    # Compute row-wise statistics
    row_means = np.nanmean(intensity_matrix, axis=1)
    row_sds = np.nanstd(intensity_matrix, axis=1, ddof=1)
    
    # Remove any NaN
    valid = ~(np.isnan(row_means) | np.isnan(row_sds))
    row_means = row_means[valid]
    row_sds = row_sds[valid]
    
    # Sort by mean for smoother
    sort_idx = np.argsort(row_means)
    means_sorted = row_means[sort_idx]
    sds_sorted = row_sds[sort_idx]
    
    # Running median smoother (approximation of loess)
    window = max(len(means_sorted) // n_bins, 5)
    smooth_sds = _running_median(sds_sorted, window)
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scattergl(
        x=row_means,
        y=row_sds,
        mode="markers",
        marker=dict(size=3, color="rgba(100, 100, 100, 0.3)"),
        name="Proteins",
        hoverinfo="skip",
    ))
    
    # Smoother line
    fig.add_trace(go.Scatter(
        x=means_sorted,
        y=smooth_sds,
        mode="lines",
        line=dict(color="red", width=3),
        name="Trend",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Mean (across samples)",
        yaxis_title="Standard Deviation",
        template="plotly_white",
        width=600,
        height=500,
    )
    
    return fig


def _running_median(arr, window):
    """Compute running median with given window size."""
    result = np.zeros_like(arr)
    half = window // 2
    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        result[i] = np.median(arr[start:end])
    return result
