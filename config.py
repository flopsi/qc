"""Global configuration: species map, thresholds, color palettes."""

# Species color mapping
SPECIES_COLORS = {
    "human": "#4C78A8",   # Blue
    "yeast": "#F58518",   # Orange
    "ecoli": "#E45756",   # Red
    "unknown": "#72B7B2", # Teal
}

# Condition color mapping
CONDITION_COLORS = {
    "A": "#4C78A8",  # Blue
    "B": "#F58518",  # Orange
}

# QC thresholds
PERMANOVA_ALPHA = 0.05
PERMANOVA_R2_GOOD = 0.3
SILHOUETTE_GOOD = 0.5
CV_THRESHOLD = 20.0  # %
ICC_EXCELLENT = 0.9
ICC_GOOD = 0.75

# Intensity bin labels
BIN_LABELS = ["Q1", "Q2", "Q3", "Q4"]

# Default number of permutations
DEFAULT_PERMUTATIONS = 999

# PCA defaults
DEFAULT_N_COMPONENTS = 5
DEFAULT_SCALE = True
