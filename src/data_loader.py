"""
Module: Load and prepare the Hillstrom uplift modeling dataset.

This module handles dataset discovery, optional balanced sampling, and feature
encoding for the Kevin Hillstrom e-mail campaign data. It is used throughout
the project to produce treatment, outcome, and feature matrices for modeling.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


RAW_FEATURE_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
CATEGORICAL_COLUMNS = ["zip_code", "channel", "history_segment"]
REQUIRED_COLUMNS = set(RAW_FEATURE_COLUMNS + ["segment", "conversion"])


def _sample_balanced_subset(
    df: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Sample a subset while preserving treatment and outcome coverage.

    Args:
        df: Full Hillstrom dataframe containing segment and conversion columns.
        n_samples: Number of rows to sample for a smaller modeling dataset.
        seed: Random seed used for deterministic sampling.

    Returns:
        A shuffled dataframe with approximately stratified treatment/outcome mix.

    Raises:
        ValueError: If the requested sample is smaller than the number of strata.
    """
    treatment = (df["segment"] != "No E-Mail").astype(int)
    strata = treatment.astype(str) + "__" + df["conversion"].astype(str)
    group_counts = strata.value_counts().sort_index()

    if n_samples < len(group_counts):
        raise ValueError("n_samples must be at least the number of treatment/outcome strata.")

    target_counts = pd.Series(1, index=group_counts.index, dtype=int)
    remaining = n_samples - len(group_counts)

    if remaining > 0:
        proportions = group_counts / group_counts.sum()
        raw_extra = proportions * remaining
        extra_counts = raw_extra.astype(int)
        target_counts = target_counts + extra_counts

        leftover = int(remaining - extra_counts.sum())
        remainders = (raw_extra - extra_counts).sort_values(ascending=False)

        for group_name in remainders.index:
            if leftover == 0:
                break
            if target_counts[group_name] < group_counts[group_name]:
                target_counts[group_name] += 1
                leftover -= 1

    sampled_parts = []
    for group_name, target_count in target_counts.items():
        group_df = df.loc[strata == group_name]
        sampled_parts.append(
            group_df.sample(
                n=min(int(target_count), len(group_df)),
                random_state=seed,
            )
        )

    sampled_df = pd.concat(sampled_parts)

    if len(sampled_df) < n_samples:
        remaining_df = df.drop(index=sampled_df.index)
        sampled_df = pd.concat(
            [
                sampled_df,
                remaining_df.sample(
                    n=n_samples - len(sampled_df),
                    random_state=seed,
                ),
            ]
        )

    return sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def find_hillstrom_csv(data_dir: Optional[Path | str] = None) -> Path:
    """
    Locate the Hillstrom CSV file in the project data directory.

    Args:
        data_dir: Optional directory to search instead of the default `data/`.

    Returns:
        Path to the first CSV whose columns match the expected Hillstrom schema.

    Raises:
        FileNotFoundError: If no compatible CSV is found in the search directory.
    """
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    csv_paths = sorted(data_dir.glob("*.csv"))

    for csv_path in csv_paths:
        columns = set(pd.read_csv(csv_path, nrows=0).columns)
        if REQUIRED_COLUMNS.issubset(columns):
            return csv_path

    raise FileNotFoundError(
        "Could not find a Hillstrom dataset CSV in the data/ directory."
    )


def load_hillstrom_dataframe(
    csv_path: Optional[Path | str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load Hillstrom data and convert it into a modeling dataframe.

    Args:
        csv_path: Optional explicit path to the Hillstrom CSV file.
        n_samples: Optional row count for a smaller sampled dataset.
        seed: Random seed used when sampling rows.

    Returns:
        A dataframe with one-hot encoded features plus `treatment` and `outcome`.
    """
    csv_path = Path(csv_path) if csv_path else find_hillstrom_csv()
    df = pd.read_csv(csv_path)

    if n_samples is not None and n_samples < len(df):
        df = _sample_balanced_subset(df, n_samples=n_samples, seed=seed)

    features = pd.get_dummies(
        df[RAW_FEATURE_COLUMNS],
        columns=CATEGORICAL_COLUMNS,
        dtype=int,
    )

    modeling_df = features.copy()
    modeling_df["treatment"] = (df["segment"] != "No E-Mail").astype(int).to_numpy()
    modeling_df["outcome"] = df["conversion"].astype(int).to_numpy()

    return modeling_df


def load_uplift_data(
    csv_path: Optional[Path | str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return Hillstrom features, treatment, and outcome ready for learners.

    Args:
        csv_path: Optional explicit path to the Hillstrom CSV file.
        n_samples: Optional row count for a smaller sampled dataset.
        seed: Random seed used when sampling rows.

    Returns:
        A tuple of `(X, T, Y)` where `X` is encoded features and `T`, `Y`
        are binary treatment and conversion series.
    """
    modeling_df = load_hillstrom_dataframe(
        csv_path=csv_path,
        n_samples=n_samples,
        seed=seed,
    )
    X = modeling_df.drop(columns=["treatment", "outcome"])
    T = modeling_df["treatment"]
    Y = modeling_df["outcome"]
    return X, T, Y
