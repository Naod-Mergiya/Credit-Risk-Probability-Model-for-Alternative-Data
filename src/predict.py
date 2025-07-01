"""Batch inference script."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.data_processing import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch predictions")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to CSV with new data")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model.pkl"))
    parser.add_argument("--out-csv", type=Path, default=Path("predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = joblib.load(args.model_path)
    df_new = pd.read_csv(args.data_path)
    X, _ = engineer_features(df_new)

    preds = model.predict_proba(X)[:, 1]
    df_new["risk_probability"] = preds
    df_new.to_csv(args.out_csv, index=False)
    logger.info("Saved predictions to %s", args.out_csv)


if __name__ == "__main__":
    main()