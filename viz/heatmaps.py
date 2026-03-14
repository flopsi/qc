"""Correlation and clustering heatmaps."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def create_correlation_heatmap(
    df: pl.DataFrame,
    intensity_cols: list[str],
    title: str = "Sample Correlation Heatmap",
) -> go.Figure:
    """Create sample-sample correlation heatmap."""
    mat = df.select(intensity_cols).to_numpy().T.astype(np.float64)
    
    # Drop proteins with NaN
    valid = ~np.isnan(mat).any(axis=0)
    mat_clean = mat[:, valid]
    
    # Compute correlation matrix
    corr = np.corrcoef(mat_clean)
    
    fig = px.imshow(
        corr,
        x=intensity_cols,
        y=intensity_cols,
        color_continuous_scale="RdBu_r",
        zmin=0.9,
        zmax=1.0,
        title=title,
        template="plotly_white",
        text_auto=".3f",
    )
    
    fig.update_layout(
        width=600,
        height=550,
    )
    
    return fig


def create_missing_value_heatmap(
    df: pl.DataFrame,
    intensity_cols: list[str],
    title: str = "Missing Value Pattern",
) -> go.Figure:
    """Create heatmap showing missing value pattern."""
    mat = df.select(intensity_cols).to_numpy()
    missing = np.isnan(mat).astype(int)
    
    # Show first 100 proteins for readability
    n_show = min(100, missing.shape[0])
    
    fig = px.imshow(
        missing[:n_show, :],
        x=intensity_cols,
        y=[f"Protein {i+1}" for i in range(n_show)],
        color_continuous_scale=["white", "red"],
        title=f"{title} (first {n_show} proteins)",
        template="plotly_white",
        aspect="auto",
    )
    
    fig.update_layout(
        width=800,
        height=600,
        coloraxis_colorbar=dict(
            title="Missing",
            tickvals=[0, 1],
            ticktext=["Present", "Missing"],
        ),
    )
    
    return fig
