from src.data_loader import load_hillstrom_dataframe, load_uplift_data


def generate_dataset(n_samples=None, seed=42, csv_path=None):
    """Backward-compatible entrypoint that now loads the real Hillstrom data."""
    return load_hillstrom_dataframe(
        csv_path=csv_path,
        n_samples=n_samples,
        seed=seed,
    )


def get_uplift_data(n_samples=None, seed=42, csv_path=None):
    return load_uplift_data(
        csv_path=csv_path,
        n_samples=n_samples,
        seed=seed,
    )
