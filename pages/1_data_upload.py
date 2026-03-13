"""Page 1 - Upload & Preview: CSV upload, column detection, session state."""
import streamlit as st
import polars as pl
import numpy as np
from config import CFG

st.header("Upload & Preview")

uploaded = st.file_uploader("Upload a CSV or TSV file", type=["csv", "tsv", "txt"])

if uploaded is not None:
    try:
        raw = uploaded.read()
        sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
        df = pl.read_csv(raw, separator=sep, infer_schema_length=5000)
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        st.stop()

    st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    # Detect intensity columns
    intensity_cols = [c for c in df.columns if c.startswith(CFG.intensity_prefix)]
    meta_cols = [c for c in df.columns if c not in intensity_cols]

    if not intensity_cols:
        st.warning(
            f"No columns starting with '{CFG.intensity_prefix}' found. "
            "Please check your column naming convention."
        )
        st.stop()

    st.info(f"Detected **{len(intensity_cols)}** intensity columns and **{len(meta_cols)}** metadata columns.")

    # Store in session state
    st.session_state["df"] = df
    st.session_state["intensity_cols"] = intensity_cols
    st.session_state["meta_cols"] = meta_cols

    # Build a long-form matrix for downstream pages
    id_cols = [c for c in meta_cols if c in df.columns]
    long = df.unpivot(
        index=id_cols,
        on=intensity_cols,
        variable_name="Sample",
        value_name="Intensity",
    )
    st.session_state["long"] = long

    # Preview
    with st.expander("Raw data preview", expanded=True):
        st.dataframe(df.head(200).to_pandas(), use_container_width=True)

    with st.expander("Intensity column list"):
        st.write(intensity_cols)

    # Basic summary stats
    with st.expander("Summary statistics"):
        int_df = df.select(intensity_cols)
        summary = int_df.describe().to_pandas()
        st.dataframe(summary, use_container_width=True)
else:
    if "df" in st.session_state:
        st.info("Using previously uploaded data.")
        st.dataframe(st.session_state["df"].head(100).to_pandas(), use_container_width=True)
    else:
        st.info("Please upload a proteomics intensity matrix to begin.")
