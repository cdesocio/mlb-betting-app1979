from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR
from .data_pipeline import FEATURE_COLUMNS

MODEL_PATH = MODELS_DIR / "moneyline_model.joblib"


@dataclass
class TrainResult:
    model_path: Path
    train_rows: int
    test_rows: int
    log_loss_value: float
    brier_value: float
    auc_value: float


def train_moneyline_model(dataset: pd.DataFrame, min_rows: int = 500) -> TrainResult:
    if dataset.empty:
        raise ValueError("Training dataset is empty.")
    if len(dataset) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows to train reliably. Found {len(dataset)}.")

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["home_win"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    log_loss_value = float(log_loss(y_test, probabilities))
    brier_value = float(brier_score_loss(y_test, probabilities))
    auc_value = float(roc_auc_score(y_test, probabilities))

    joblib.dump(model, MODEL_PATH)
    return TrainResult(
        model_path=MODEL_PATH,
        train_rows=len(X_train),
        test_rows=len(X_test),
        log_loss_value=log_loss_value,
        brier_value=brier_value,
        auc_value=auc_value,
    )


def load_moneyline_model(model_path: Path | None = None):
    chosen = model_path or MODEL_PATH
    if not chosen.exists():
        raise FileNotFoundError(
            f"Model file not found at {chosen}. Run the training script first."
        )
    return joblib.load(chosen)


def predict_home_win_probability(model, frame: pd.DataFrame) -> pd.Series:
    probs = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    return pd.Series(np.clip(probs, 0.01, 0.99), index=frame.index, name="model_home_win_prob")
