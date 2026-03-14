"""PVCA bar chart visualization."""
import plotly.express as px
import plotly.graph_objects as go
from analysis.variance_components import PVCAResult


def create_pvca_bar(
    pvca_result: PVCAResult,
    title: str = "Variance Components (PVCA)",
) -> go.Figure:
    """Create bar chart of variance components."""
    components = pvca_result.components
    
    fig = px.bar(
        x=list(components.keys()),
        y=[v * 100 for v in components.values()],
        labels={"x": "Source", "y": "Variance Explained (%)"},
        title=f"{title} — {pvca_result.n_pcs_used} PCs retained "
              f"(≥{pvca_result.cumulative_variance_threshold*100:.0f}% variance)",
        template="plotly_white",
        color=list(components.keys()),
        color_discrete_sequence=["#4C78A8", "#F58518", "#72B7B2"],
    )
    
    fig.update_layout(
        showlegend=False,
        width=600,
        height=500,
        yaxis_range=[0, 100],
    )
    
    # Add value labels on bars
    for i, (name, val) in enumerate(components.items()):
        fig.add_annotation(
            x=name,
            y=val * 100 + 2,
            text=f"{val*100:.1f}%",
            showarrow=False,
            font=dict(size=14, color="black"),
        )
    
    return fig
