"""Page 3 - Intensity Bins: bin proteins by log2 intensity, PCA per bin."""
import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import CFG

st.header("Intensity Bins")

if "df" not in st.session_state:
    st.warning("Please upload data on the Upload & Preview page first.")
    st.stop()

df = st.session_state["df"]
intensity_cols = st.session_state["intensity_cols"]

mat = df.select(intensity_cols).to_numpy().astype(np.float64)
mean_intensity = np.nanmean(mat, axis=1)
log2_mean = np.log2(np.where(mean_intensity > 0, mean_intensity, np.nan))

edges = list(CFG.bin_edges)
labels = list(CFG.bin_labels)

bins = np.digitize(log2_mean, edges) - 1
bins = np.clip(bins, 0, len(labels) - 1)
bin_names = [labels[b] if np.isfinite(log2_mean[i]) else "Missing" for i, b in enumerate(bins)]

df_binned = df.with_columns(pl.Series("IntensityBin", bin_names))

# Distribution of bins
st.subheader("Protein count per intensity bin")
bin_counts = df_binned.group_by("IntensityBin").len().sort("IntensityBin").to_pandas()
fig_bar = px.bar(bin_counts, x="IntensityBin", y="len", labels={"len": "Count"}, title="Proteins per bin")
st.plotly_chart(fig_bar, use_container_width=True)

# PCA per bin
st.subheader("PCA per intensity bin")
color_col = st.selectbox("Color by", [None] + st.session_state["meta_cols"], key="bin_color")

for label in labels:
    mask = np.array(bin_names) == label
    n_in_bin = mask.sum()
    if n_in_bin < 3:
        continue

    with st.expander(f"Bin {label} ({n_in_bin} proteins)", expanded=False):
        sub_mat = mat[mask]
        sub_mat = np.where(np.isfinite(sub_mat), sub_mat, 0)

        n_comp = min(3, sub_mat.shape[0], sub_mat.shape[1])
        if n_comp < 2:
            st.info("Not enough data for PCA in this bin.")
            continue

        scaler = StandardScaler()
        sub_scaled = scaler.fit_transform(sub_mat.T)  # samples as rows
        pca = PCA(n_components=min(2, sub_scaled.shape[1]))
        scores = pca.fit_transform(sub_scaled)

        score_df = pl.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1] if scores.shape[1] > 1 else np.zeros(scores.shape[0])})
        score_df = score_df.with_columns(pl.Series("Sample", intensity_cols))

        pd_scores = score_df.to_pandas()
        fig = px.scatter(
            pd_scores, x="PC1", y="PC2", text="Sample",
            title=f"PCA - Bin {label}",
            labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if len(pca.explained_variance_ratio_) > 1 else "PC2"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
