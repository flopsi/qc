"""Proteomics QC Dashboard - Configuration."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QCConfig:
    """Central configuration for the QC dashboard."""
    # Column naming conventions
    intensity_prefix: str = "Intensity "
    condition_col: str = "Condition"
    replicate_col: str = "Replicate"
    species_col: str = "Species"
    protein_col: str = "Protein"

    # Intensity bin edges (log2 scale)
    bin_edges: tuple = (0, 10, 15, 20, 25, 35)
    bin_labels: tuple = ("<10", "10-15", "15-20", "20-25", ">25")

    # PCA defaults
    pca_n_components: int = 5
    pca_scale: bool = True

    # PERMANOVA defaults
    permanova_permutations: int = 999

    # CV thresholds
    cv_good: float = 0.10
    cv_acceptable: float = 0.20

    # Plotly color palette
    color_discrete_sequence: tuple = (
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    )


CFG = QCConfig()
