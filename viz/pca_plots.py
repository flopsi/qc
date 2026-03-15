"""Interactive PCA scatter plots using Plotly."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analysis.pca_engine import PCAResult


def create_pca_scatter(
    pca_result: PCAResult,
    conditions: list[str],
    pc_x: int = 1,
    pc_y: int = 2,
    title: str = "PCA",
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Create interactive PCA scatter plot with confidence ellipses.
    
    Args:
        pca_result: PCAResult from run_pca
        conditions: Condition label for each sample
        pc_x: PC for x-axis (1-indexed)
        pc_y: PC for y-axis (1-indexed)
        title: Plot title
        color_map: Optional color mapping for conditions
    """
    import pandas as pd
    
    scores_df = pd.DataFrame({
        f"PC{pc_x}": pca_result.scores[:, pc_x - 1],
        f"PC{pc_y}": pca_result.scores[:, pc_y - 1],
        "Condition": conditions,
        "Sample": pca_result.sample_names,
    })
    
    var_x = pca_result.explained_variance[pc_x - 1] * 100
    var_y = pca_result.explained_variance[pc_y - 1] * 100
    
    fig = px.scatter(
        scores_df,
        x=f"PC{pc_x}",
        y=f"PC{pc_y}",
        color="Condition",
        hover_data=["Sample"],
        labels={
            f"PC{pc_x}": f"PC{pc_x} ({var_x:.1f}%)",
            f"PC{pc_y}": f"PC{pc_y} ({var_y:.1f}%)",
        },
        title=title,
        template="plotly_white",
        color_discrete_map=color_map,
    )
    
    fig.update_traces(marker=dict(size=14, line=dict(width=2, color="white")))
    
    # Add 95% confidence ellipses
    for cond in dict.fromkeys(conditions):
        mask = [c == cond for c in conditions]
        x_vals = pca_result.scores[mask, pc_x - 1]
        y_vals = pca_result.scores[mask, pc_y - 1]
        
        if len(x_vals) >= 3:
            _add_confidence_ellipse(fig, x_vals, y_vals, cond, color_map)
    
    fig.update_layout(
        width=800,
        height=600,
        legend=dict(font=dict(size=14)),
    )
    
    return fig


def _add_confidence_ellipse(fig, x, y, name, color_map=None):
    """Add 95% confidence ellipse to figure."""

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues/vectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvalues = np.maximum(eigenvalues, 0)  # Guard against negative from numerical noise
    eigenvectors = eigenvectors[:, order]
    
    # 95% confidence interval chi-squared value for 2 DOF
    chi2_val = 5.991
    
    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, 100)
    a = np.sqrt(eigenvalues[0] * chi2_val)
    b = np.sqrt(eigenvalues[1] * chi2_val)
    
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    ellipse_x = a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle) + mean_x
    ellipse_y = a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle) + mean_y
    
    color = None
    if color_map and name in color_map:
        color = color_map[name]
    
    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode="lines",
        line=dict(dash="dash", width=1, color=color),
        showlegend=False,
        hoverinfo="skip",
    ))


def create_scree_plot(
    pca_result: PCAResult,
    title: str = "Scree Plot",
) -> go.Figure:
    """Create variance explained bar chart."""
    fig = px.bar(
        x=[f"PC{i+1}" for i in range(pca_result.n_components)],
        y=pca_result.explained_variance * 100,
        labels={"x": "Component", "y": "Variance Explained (%)"},
        title=title,
        template="plotly_white",
    )
    fig.update_traces(marker_color="#4C78A8")
    return fig
