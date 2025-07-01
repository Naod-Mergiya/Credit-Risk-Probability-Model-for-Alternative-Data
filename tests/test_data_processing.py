from pathlib import Path

import pandas as pd
from src.data_processing import engineer_features


def test_engineer_features(tmp_path: Path):
    # Create sample dataframe
    df = pd.DataFrame({
        "FraudResult": [0, 1],
        "Amount": [100.0, 200.0],
    })

    X, y = engineer_features(df)

    assert "Amount" in X.columns
    assert len(X) == len(y) == 2