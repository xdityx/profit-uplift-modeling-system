"""
Module: Preserve the historical dataset-loading entrypoints.

This module keeps the old simulation API stable while routing calls to the real
Hillstrom dataset loader. It lets downstream code continue using familiar
function names even though the project no longer relies on synthetic data.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_loader import load_hillstrom_dataframe, load_uplift_data


def generate_dataset(
    n_samples: Optional[int] = None,
    seed: int = 42,
    csv_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Return a modeling dataframe through the legacy simulation entrypoint.

    Args:
        n_samples: Optional number of rows to sample from the dataset.
        seed: Random seed used for deterministic sampling.
        csv_path: Optional explicit path to the Hillstrom CSV file.

    Returns:
        A dataframe of encoded features plus `treatment` and `outcome`.
    """
    return load_hillstrom_dataframe(
        csv_path=csv_path,
        n_samples=n_samples,
        seed=seed,
    )


def get_uplift_data(
    n_samples: Optional[int] = None,
    seed: int = 42,
    csv_path: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return features, treatment, and outcome for uplift estimators.

    Args:
        n_samples: Optional number of rows to sample from the dataset.
        seed: Random seed used for deterministic sampling.
        csv_path: Optional explicit path to the Hillstrom CSV file.

    Returns:
        A tuple of `(X, T, Y)` ready for propensity and uplift models.
    """
    return load_uplift_data(
        csv_path=csv_path,
        n_samples=n_samples,
        seed=seed,
    )
