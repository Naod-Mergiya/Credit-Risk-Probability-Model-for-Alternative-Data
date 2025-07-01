"""Feature engineering pipeline for the credit-risk model.

This module defines a reusable scikit-learn ``Pipeline`` that converts the raw
transaction-level dataframe into a model-ready feature matrix.

Key capabilities
----------------
1. Aggregate per-customer statistics – total, mean, count and standard
   deviation of transaction amounts.
2. Datetime extraction – hour, day, month and year from the transaction
   timestamp.
3. Missing-value handling – numerical columns imputed with median, categorical
   with the string ``"missing"``.
4. Encoding & scaling – categorical variables one-hot encoded, numerical
   variables standardised.
5. Optional Weight-of-Evidence (WoE) encoding – if ``xverse`` is available a
   WoE transformer is appended as the final step to improve monotonicity and
   interpretability for credit-risk modelling.

The final output is a dense ``numpy.ndarray`` ready for model consumption.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xverse.transformer import WOETransformer  # type: ignore
except ModuleNotFoundError:  # optional dependency
    WOETransformer = None  # type: ignore

# ---------------------------------------------------------------------------
# Column name constants – adjust if your raw schema differs
# ---------------------------------------------------------------------------
ID_COL: str = "CustomerId"
DATETIME_COL: str = "TransactionDate"
AMOUNT_COL: str = "Amount"
TARGET: str = "FraudResult"  # not used in transforms but exposed for callers

# Output of aggregation step will add these suffixes to ``AMOUNT_COL``
AGG_SUFFIXES: List[str] = ["_sum", "_mean", "_count", "_std"]

# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------
class Aggregator(BaseEstimator, TransformerMixin):
    """Compute per-customer aggregate statistics and merge back to rows."""

    def __init__(self, id_col: str | None = None, amount_col: str | None = None):
        """Initialise with optional column overrides.

        Setting *id_col* or *amount_col* to ``None`` (default) means the
        transformer will fall back to the module-level constants ``ID_COL`` and
        ``AMOUNT_COL``. This allows notebooks to change the global constants at
        runtime and still have the pipeline pick them up without rebuilding the
        entire class definition.
        """
        # Allow overriding after import time
        self.id_col = id_col or ID_COL
        self.amount_col = amount_col or AMOUNT_COL
        self._feature_names: list[str] = []

    # pylint: disable=unused-argument
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # type: ignore[override]
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Aggregator requires a pandas DataFrame input")

        aggs = (
            X.groupby(self.id_col)[self.amount_col]
            .agg(["sum", "mean", "count", "std"])
            .rename(
                columns={
                    "sum": f"{self.amount_col}_sum",
                    "mean": f"{self.amount_col}_mean",
                    "count": f"{self.amount_col}_count",
                    "std": f"{self.amount_col}_std",
                }
            )
        )

        self._aggregates_ = aggs  # type: ignore[attr-defined]
        self._feature_names = aggs.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        if not hasattr(self, "_aggregates_"):
            raise RuntimeError("Must call fit before transform on Aggregator")
        # Merge aggregates but **keep all original columns** so downstream
        # transformers (e.g. DatetimeExtractor, ColumnTransformer) still have
        # access to them.
        out = X.merge(
            self._aggregates_, how="left", left_on=self.id_col, right_index=True
        )
        return out

    # scikit-learn 1.3+ compatible feature names
    def get_feature_names_out(self, input_features: list[str] | None = None):  # noqa: N802
        return np.array(self._feature_names)


class DatetimeExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal components from a datetime column."""

    def __init__(self, datetime_col: str | None = None):
        """Initialise with optional datetime column override.

        If *datetime_col* is not provided the transformer uses the module-level
        ``DATETIME_COL`` variable, which can itself be monkey-patched at
        runtime (e.g. in notebooks) *before* building the pipeline.
        """
        # Use override if provided, else fall back to global constant
        self.datetime_col = datetime_col or DATETIME_COL
        self._feature_names: list[str] = [
            f"{self.datetime_col}_hour",
            f"{self.datetime_col}_day",
            f"{self.datetime_col}_month",
            f"{self.datetime_col}_year",
        ]

    # pylint: disable=unused-argument
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # type: ignore[override]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")
        new_cols = pd.DataFrame(
            {
                f"{self.datetime_col}_hour": dt.dt.hour.astype("Int16"),
                f"{self.datetime_col}_day": dt.dt.day.astype("Int16"),
                f"{self.datetime_col}_month": dt.dt.month.astype("Int16"),
                f"{self.datetime_col}_year": dt.dt.year.astype("Int16"),
            },
            index=X.index,
        )
        # Preserve original columns so downstream transformers (e.g. Aggregator)
        # still have access to the raw datetime and other fields.
        return pd.concat([X, new_cols], axis=1)

    def get_feature_names_out(self, input_features: list[str] | None = None):  # noqa: N802
        return np.array(self._feature_names)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Return a complete ``Pipeline`` performing all feature engineering steps."""

    # Pre-compute feature groups created by Aggregator & DatetimeExtractor
    numeric_features = [
        f"{AMOUNT_COL}_sum",
        f"{AMOUNT_COL}_mean",
        f"{AMOUNT_COL}_count",
        f"{AMOUNT_COL}_std",
    ]
    datetime_features = [
        f"{DATETIME_COL}_hour",
        f"{DATETIME_COL}_day",
        f"{DATETIME_COL}_month",
        f"{DATETIME_COL}_year",
    ]

    # Example categorical list – extend as needed
    categorical_features: list[str] = [
        "TransactionType",
        "Channel",
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric_features + datetime_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    steps: list[tuple[str, BaseEstimator]] = [
        ("datetime", DatetimeExtractor()),  # extract first while column exists
        ("aggregate", Aggregator()),        # add aggregate stats, keep columns
        ("prep", preprocessor),
    ]

    # Optionally append WoE encoding if xverse is installed
    if WOETransformer is not None:
        steps.append(("woe", WOETransformer(features_to_encode="auto")))

    return Pipeline(steps)


# Convenience lists so notebooks can access engineered column names easily
AGG_COLS = [f"{AMOUNT_COL}{suf}" for suf in AGG_SUFFIXES]
FEATURES: List[str] = [ID_COL, DATETIME_COL, AMOUNT_COL, *AGG_COLS]