"""Page 4 - Species PCA: filter by species column and run PCA per species."""
import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import CFG

st.header("Species PCA")

if "df" not in st.session_state:
    st.warning("Please upload data on the Upload & Preview page first.")
    st.stop()

df = st.session_state["df"]
intensity_cols = st.session_state["intensity_cols"]

if CFG.species_col not in df.columns:
    st.warning(f"No '{CFG.species_col}' column found in data. This page requires a species annotation column.")
    st.stop()

species_list = df[CFG.species_col].unique().sort().to_list()
st.info(f"Found {len(species_list)} species: {', '.join(str(s) for s in species_list)}")

color_col = st.selectbox("Color by", [None] + st.session_state["meta_cols"], key="sp_color")

for species in species_list:
    sp_df = df.filter(pl.col(CFG.species_col) == species)
    n_proteins = sp_df.shape[0]

    if n_proteins < 3:
        st.info(f"Species '{species}': only {n_proteins} proteins, skipping PCA.")
        continue

    with st.expander(f"{species} ({n_proteins} proteins)", expanded=(len(species_list) <= 3)):
        mat = sp_df.select(intensity_cols).to_numpy().astype(np.float64)
        mat = np.where(np.isfinite(mat), mat, 0)

        # Transpose: samples as rows
        mat_t = mat.T
        n_comp = min(3, mat_t.shape[0], mat_t.shape[1])
        if n_comp < 2:
            st.info("Not enough dimensions for PCA.")
            continue

        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat_t)
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(mat_scaled)

        score_df = pl.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
        score_df = score_df.with_columns(pl.Series("Sample", intensity_cols))

        pd_scores = score_df.to_pandas()
        fig = px.scatter(
            pd_scores, x="PC1", y="PC2", text="Sample",
            title=f"PCA - {species}",
            labels={
                "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            },
            color_discrete_sequence=list(CFG.color_discrete_sequence),
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        # Variance explained
        col1, col2 = st.columns(2)
        col1.metric("PC1 variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
        col2.metric("PC2 variance", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
