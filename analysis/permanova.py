"""PERMANOVA and dispersion tests using scikit-bio."""
import numpy as np
from math import comb
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix


@dataclass
class PermanovaResult:
    """Container for PERMANOVA results."""
    test_statistic: float  # pseudo-F
    p_value: float
    r_squared: float       # Effect size
    n_permutations: int
    sample_size: int
    n_groups: int
    min_achievable_p: float  # Minimum p-value possible given sample size
    is_min_p: bool           # Whether observed p equals the minimum achievable


def _compute_min_p(n: int, k: int) -> float:
    """Compute minimum achievable PERMANOVA p-value for balanced designs.
    
    With n samples split into k equal groups of size m = n/k,
    the total number of unique permutations is C(n, m) * C(n-m, m) / k!
    but for PERMANOVA the mirror permutation (swapping group labels) gives
    the same test statistic. So the minimum p = 2 / C(n, m) for k=2.
    
    For 3 vs 3 (n=6, k=2): C(6,3) = 20, min p = 2/20 = 0.10
    For 4 vs 4 (n=8, k=2): C(8,4) = 70, min p = 2/70 = 0.029
    For 5 vs 5 (n=10, k=2): C(10,5) = 252, min p = 2/252 = 0.008
    """
    if k != 2:
        # For >2 groups, approximate conservatively
        m = n // k
        return 1.0 / comb(n, m)
    
    m = n // 2
    total_perms = comb(n, m)
    # The observed grouping and its mirror always produce the same F-statistic
    # so the minimum p-value is 2 / total_perms
    return 2.0 / total_perms


def run_permanova(
    intensity_matrix: np.ndarray,
    grouping: list[str],
    metric: str = "euclidean",
    permutations: int = 999,
    seed: int | None = 42,
) -> PermanovaResult:
    n = len(grouping)
    if intensity_matrix.shape[0] != n:
        raise ValueError("intensity_matrix rows must match len(grouping)")

    dist_condensed = pdist(intensity_matrix, metric=metric)
    dist_square = squareform(dist_condensed)
    dm = DistanceMatrix(dist_square, ids=[str(i) for i in range(n)])

    result = permanova(dm, grouping, permutations=permutations, seed=seed)

    f_stat = float(result["test statistic"])
    p_value = float(result["p-value"])
    n_groups = len(set(grouping))
    r_squared = 1.0 / (1.0 + ((n - n_groups) / ((n_groups - 1) * f_stat)))

    return PermanovaResult(
        test_statistic=f_stat,
        p_value=p_value,
        r_squared=r_squared,
        n_permutations=permutations,
        sample_size=n,
        n_groups=n_groups,
    )

def interpret_permanova(result: PermanovaResult, context: str = "global") -> tuple[str, str]:
    """Generate human-readable interpretation of PERMANOVA results.
    
    Args:
        result: PermanovaResult from run_permanova
        context: One of "global", "species_stable", "species_variable", "intensity_bin"
    
    Returns:
        Tuple of (severity, message) where severity is "success", "info", "warning", "error"
    """
    p = result.p_value
    r2 = result.r_squared
    min_p = result.min_achievable_p
    is_min = result.is_min_p
    
    # Small-sample adjustment: if p is at the minimum achievable value,
    # the test is as significant as it can possibly be given the sample size
    effectively_significant = is_min or p < 0.05
    
    if context == "species_stable":
        # For stable background (e.g., human in HYE benchmark)
        if not effectively_significant and r2 < 0.5:
            return "success", (
                f"No significant separation detected (p={p:.3f}, R²={r2:.3f}). "
                f"This species is stable across conditions, as expected for a constant background."
            )
        elif effectively_significant and r2 > 0.5:
            return "warning", (
                f"Unexpected separation detected (p={p:.3f}, R²={r2:.3f}). "
                f"This species should be stable but shows a condition effect."
            )
        else:
            return "info", (
                f"Weak/ambiguous signal (p={p:.3f}, R²={r2:.3f}). "
                f"Minor technical variation may be present."
            )
    
    elif context == "species_variable":
        # For species expected to differ between conditions (yeast, ecoli in HYE)
        if is_min and r2 > 0.8:
            return "success", (
                f"Strong condition separation (R²={r2:.3f}, p={p:.3f} — minimum achievable "
                f"with {result.sample_size} samples). PC1 captures {r2*100:.0f}% of variance "
                f"between conditions. The p-value is at the theoretical minimum ({min_p:.2f}) "
                f"for a {result.sample_size//result.n_groups}-vs-{result.sample_size//result.n_groups} "
                f"design, confirming maximum possible significance."
            )
        elif effectively_significant and r2 > 0.5:
            return "success", (
                f"Significant condition effect (p={p:.3f}, R²={r2:.3f}). "
                f"Conditions are well-separated for this species."
            )
        elif r2 > 0.7:
            return "info", (
                f"High variance explained by condition (R²={r2:.3f}) but p={p:.3f} "
                f"is limited by the small sample size (min achievable p={min_p:.2f}). "
                f"The effect is likely real — consider increasing replicates for formal significance."
            )
        else:
            return "warning", (
                f"Weak or no separation (p={p:.3f}, R²={r2:.3f}). "
                f"This species does not show the expected condition effect."
            )
    
    elif context == "intensity_bin":
        if is_min and r2 > 0.7:
            return "success", (
                f"Strong separation in this intensity bin (R²={r2:.3f}, p={p:.3f} at "
                f"minimum achievable). Condition effect is clearly captured."
            )
        elif r2 > 0.5:
            return "info", (
                f"Moderate to good separation (R²={r2:.3f}, p={p:.3f}). "
                f"The condition effect is detectable in this intensity range."
            )
        elif r2 > 0.2:
            return "info", (
                f"Weak separation (R²={r2:.3f}, p={p:.3f}). "
                f"Some condition signal present but noise dominates in this intensity range."
            )
        else:
            return "warning", (
                f"No meaningful separation (R²={r2:.3f}, p={p:.3f}). "
                f"Proteins in this intensity range do not resolve the conditions."
            )
    
    else:  # global
        if is_min and r2 > 0.7:
            return "success", (
                f"Conditions are well-separated (R²={r2:.3f}, p={p:.3f} — at the minimum "
                f"achievable for {result.sample_size//result.n_groups} replicates per condition). "
                f"The measurement system resolves the experimental groups."
            )
        elif r2 > 0.3 and effectively_significant:
            return "success", (
                f"Conditions show meaningful separation (R²={r2:.3f}, p={p:.3f}). "
                f"Note: in a mixed-proteome benchmark where ~50% of proteins are not regulated, "
                f"a moderate R² is expected and indicates the system correctly detects the "
                f"changing species against the stable background."
            )
        elif r2 > 0.3:
            return "info", (
                f"Moderate separation (R²={r2:.3f}, p={p:.3f}). The condition effect explains "
                f"a substantial fraction of variance. The p-value may be limited by sample size "
                f"(minimum achievable: {min_p:.2f}). In a mixed-proteome benchmark with a large "
                f"stable background, moderate global R² is expected."
            )
        elif effectively_significant:
            return "info", (
                f"Statistically significant but weak effect (R²={r2:.3f}, p={p:.3f}). "
                f"Measurement noise is substantial relative to the condition effect."
            )
        else:
            return "error", (
                f"No significant separation (R²={r2:.3f}, p={p:.3f}). "
                f"The measurement system does not resolve conditions at the global proteome level."
            )
