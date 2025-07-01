"""Data processing and feature engineering utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV file.

    Parameters
    ----------
    path: str | Path
        Location of the raw CSV file.
    """
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", *df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Basic feature engineering.

    For now, only splits target `FraudResult` from features.
    Replace / extend with your own logic.
    """
    logger.info("Starting feature engineering")
    y = df["FraudResult"].astype(int)
    X = df.drop(columns=["FraudResult"])
    logger.info("Generated feature matrix with shape %s", X.shape)
    return X, y