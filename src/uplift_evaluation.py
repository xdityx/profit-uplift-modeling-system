import numpy as np

def qini_curve(y, treatment, uplift_scores, n_bins=10):
    """
    Compute Qini curve values.

    y: outcomes
    treatment: treatment indicator
    uplift_scores: predicted uplift
    n_bins: number of segments
    """
    order = np.argsort(-uplift_scores)
    y = y[order]
    t = treatment[order]

    cum_treated = np.cumsum(t)
    cum_outcomes_treated = np.cumsum(y * t)
    cum_outcomes_control = np.cumsum(y * (1 - t))

    gains = cum_outcomes_treated - cum_outcomes_control * (cum_treated / (np.arange(len(t)) + 1))

    return gains


def auuc(y, treatment, uplift_scores, n_bins=10):
    """Compute the area under the Qini-style uplift curve."""
    gains = qini_curve(y, treatment, uplift_scores, n_bins=n_bins)
    return float(np.trapezoid(gains))
