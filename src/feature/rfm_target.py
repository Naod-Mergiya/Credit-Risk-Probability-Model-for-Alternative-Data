"""Proxy target engineering using RFM + K-Means.

Adds a binary column ``is_high_risk`` that flags the customer cluster with the
lowest engagement (low frequency & low monetary spend).
"""
from __future__ import annotations

from typing import Hashable, Literal

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

__all__ = [
    "add_rfm_target",
]


def _compute_rfm(
    df: pd.DataFrame,
    *,
    id_col: Hashable,
    amount_col: Hashable,
    datetime_col: Hashable,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return an RFM frame indexed by ``id_col``."""
    rfm = (
        df.groupby(id_col).agg(
            Recency=(datetime_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(datetime_col, "count"),
            Monetary=(amount_col, "sum"),
        )
    )
    return rfm.astype("float64")


def add_rfm_target(
    df: pd.DataFrame,
    *,
    id_col: str = "CustomerId",
    amount_col: str = "Amount",
    datetime_col: str = "TransactionStartTime",
    snapshot_date: pd.Timestamp | str | None = None,
    n_clusters: Literal[3] | int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of *df* with a new ``is_high_risk`` column.

    The high-risk cluster is chosen as the one whose centre has the smallest
    *Frequency* and *Monetary* values (ties broken by the larger *Recency*).
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    out = df.copy()
    # parse datetimes and drop any timezone info (keep them tz-naive)
    out[datetime_col] = (
        pd.to_datetime(out[datetime_col], errors="coerce")
        .dt.tz_convert(None)
    )
    if out[datetime_col].isna().all():
        raise ValueError("datetime_col could not be parsed to datetime")

    if snapshot_date is None:
        snapshot_date = out[datetime_col].max() + pd.Timedelta(days=1)
    snapshot_date = pd.to_datetime(snapshot_date)

    # RFM matrix
    rfm = _compute_rfm(
        out,
        id_col=id_col,
        amount_col=amount_col,
        datetime_col=datetime_col,
        snapshot_date=snapshot_date,
    )

    # scale & cluster
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    rfm["cluster"] = km.fit_predict(rfm_scaled)

    # pick high-risk cluster – lowest Freq + Monetary, highest Recency
    centers = pd.DataFrame(km.cluster_centers_, columns=rfm.columns[:-1])
    centers["_score"] = (
        centers["Frequency"].rank(method="average")
        + centers["Monetary"].rank(method="average")
        - centers["Recency"].rank(method="average")
    )
    risk_cluster: int = centers["_score"].idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == risk_cluster).astype("int8")

    # merge back
    out = out.merge(rfm[["is_high_risk"]], left_on=id_col, right_index=True, how="left")
    return out