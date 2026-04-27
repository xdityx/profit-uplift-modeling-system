"""
Module: Evaluate uplift model ranking quality.

This module computes Qini-style cumulative gains and summary area metrics for
uplift predictions. It is used to compare whether treatment-effect rankings
surface customers with greater incremental conversion value.
"""

import numpy as np

def qini_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Compute cumulative Qini gains for ranked uplift predictions.

    Args:
        y: Binary outcome array.
        treatment: Binary treatment indicator array.
        uplift_scores: Predicted uplift scores used for ranking.
        n_bins: Unused placeholder for future binning support.

    Returns:
        A NumPy array of cumulative gains ordered by predicted uplift.
    """
    order = np.argsort(-uplift_scores)
    y = y[order]
    t = treatment[order]

    cum_treated = np.cumsum(t)
    cum_outcomes_treated = np.cumsum(y * t)
    cum_outcomes_control = np.cumsum(y * (1 - t))

    gains = cum_outcomes_treated - cum_outcomes_control * (cum_treated / (np.arange(len(t)) + 1))

    return gains


def auuc(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Summarize uplift ranking quality as area under the Qini curve.

    Args:
        y: Binary outcome array.
        treatment: Binary treatment indicator array.
        uplift_scores: Predicted uplift scores used for ranking.
        n_bins: Unused placeholder kept for API compatibility.

    Returns:
        The scalar area under the cumulative uplift curve.
    """
    gains = qini_curve(y, treatment, uplift_scores, n_bins=n_bins)
    return float(np.trapezoid(gains))
