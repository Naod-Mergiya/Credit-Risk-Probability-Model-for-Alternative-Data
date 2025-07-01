"""Utility functions to streamline Exploratory Data Analysis (EDA)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


###############################################################################
# I/O
###############################################################################

def load_dataset(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load CSV or Excel dataset given a single path.

    Parameters
    ----------
    path: str | Path
        Filepath to CSV/XLSX.
    nrows: int | None
        Optionally limit number of rows (handy for quick iteration).
    """
    path = Path(path)
    logger.info("Loading dataset from %s", path)
    if path.suffix in {".csv"}:
        df = pd.read_csv(path, nrows=nrows)
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


###############################################################################
# Summary helpers
###############################################################################

def quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic summary (dtype, non-null %, unique #) for each column."""
    return (
        pd.DataFrame({
            "dtype": df.dtypes,
            "n_unique": df.nunique(),
            "missing_pct": df.isna().mean().mul(100).round(2),
        })
        .sort_values("missing_pct", ascending=False)
    )


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper around df.describe for numeric cols only."""
    num_cols = df.select_dtypes("number").columns
    return df[num_cols].describe().T


###############################################################################
# Visualization helpers
###############################################################################

def plot_num_distributions(df: pd.DataFrame, cols: Sequence[str] | None = None, bins: int = 30) -> None:
    """Plot histograms for numeric columns (or provided subset)."""
    if cols is None:
        cols = df.select_dtypes("number").columns
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=bins, ax=ax, kde=True)
        ax.set_title(col)
    plt.tight_layout()


def plot_cat_distributions(df: pd.DataFrame, cols: Sequence[str] | None = None, top_n: int = 15) -> None:
    """Bar charts for categorical columns (top_n most frequent)."""
    if cols is None:
        cols = df.select_dtypes(exclude="number").columns
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts().nlargest(top_n)
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_title(col)
    plt.tight_layout()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Display correlation heatmap for numeric variables."""
    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()


def boxplot_outliers(df: pd.DataFrame, cols: Sequence[str] | None = None) -> None:
    """Boxplots to visually inspect outliers per numeric column."""
    if cols is None:
        cols = df.select_dtypes("number").columns
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    