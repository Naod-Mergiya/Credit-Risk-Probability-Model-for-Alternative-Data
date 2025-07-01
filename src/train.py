"""Model training script.

Usage:
    python -m src.train --raw-path data/raw/transactions.csv --model-out artifacts/model.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_processing import engineer_features, load_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit-risk model")
    parser.add_argument("--raw-path", type=Path, required=True, help="Path to raw CSV data")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/model.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_raw = load_raw(args.raw_path)
    X, y = engineer_features(df_raw)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    logger.info("Validation ROC-AUC = %.4f", auc)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    logger.info("Saved model to %s", args.model_out)


if __name__ == "__main__":
    main()