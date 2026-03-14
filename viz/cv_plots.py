"""CV violin/box plots per intensity bin."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import polars as pl


def create_cv_violin_plot(
    cv_df: pl.DataFrame,
    conditions: list[str] | None = None,
    group_by: str = "intensity_bin",
    title: str = "CV Distribution",
) -> go.Figure:
    """Create violin plot of CV distributions.
    
    Args:
        cv_df: DataFrame from compute_cvs with cv_A, cv_B columns
        conditions: Condition labels
        group_by: Column to group by (intensity_bin or species)
        title: Plot title
    """
    if conditions is None:
        conditions = ["A", "B"]
    
    import pandas as pd
    
    # Melt to long format
    rows = []
    for cond in conditions:
        cv_col = f"cv_{cond}"
        if cv_col not in cv_df.columns:
            continue
        
        for row in cv_df.iter_rows(named=True):
            val = row.get(cv_col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            entry = {
                "CV (%)": val,
                "Condition": cond,
            }
            if group_by in row:
                entry[group_by] = row[group_by]
            rows.append(entry)
    
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No CV data available", showarrow=False)
        return fig
    
    long_df = pd.DataFrame(rows)
    
    if group_by in long_df.columns:
        fig = px.violin(
            long_df,
            x=group_by,
            y="CV (%)",
            color="Condition",
            box=True,
            points=False,
            title=title,
            template="plotly_white",
        )
    else:
        fig = px.violin(
            long_df,
            x="Condition",
            y="CV (%)",
            color="Condition",
            box=True,
            points=False,
            title=title,
            template="plotly_white",
        )
    
    fig.update_layout(width=800, height=500)
    return fig


def create_cv_summary_table(
    cv_df: pl.DataFrame,
    conditions: list[str] | None = None,
) -> pl.DataFrame:
    """Create summary statistics of CVs per condition."""
    if conditions is None:
        conditions = ["A", "B"]
    
    rows = []
    for cond in conditions:
        cv_col = f"cv_{cond}"
        if cv_col not in cv_df.columns:
            continue
        vals = cv_df[cv_col].drop_nulls().to_numpy()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        rows.append({
            "Condition": cond,
            "Median CV (%)": round(float(np.median(vals)), 2),
            "Mean CV (%)": round(float(np.mean(vals)), 2),
            "CV < 20% (%)": round(float(np.sum(vals < 20) / len(vals) * 100), 1),
            "N proteins": len(vals),
        })
    
    return pl.DataFrame(rows)
