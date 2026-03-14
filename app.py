"""Proteomics QC Dashboard — Entrypoint with st.navigation."""
import streamlit as st

st.set_page_config(
    page_title="Proteomics QC Dashboard",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define pages with sections using st.navigation
pages = {
    "Data": [
        st.Page("pages/1_data_upload.py",
                title="Upload & Preview", icon=":material/upload_file:"),
    ],
    "QC Analysis": [
        st.Page("pages/2_qc_overview.py",
                title="Global PCA", icon=":material/scatter_plot:"),
        st.Page("pages/3_intensity_bins.py",
                title="Intensity Bins", icon=":material/bar_chart:"),
        st.Page("pages/4_species_pca.py",
                title="Species PCA", icon=":material/biotech:"),
    ],
    "Transformations": [
        st.Page("pages/5_transforms.py",
                title="log2 vs glog", icon=":material/transform:"),
    ],
    "Advanced": [
        st.Page("pages/6_variance_components.py",
                title="Variance Components", icon=":material/pie_chart:"),
        st.Page("pages/7_advanced_metrics.py",
                title="ICC & CV Metrics", icon=":material/query_stats:"),
    ],
}

pg = st.navigation(pages)
pg.run()
