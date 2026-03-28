from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    odds_regions: str = os.getenv("ODDS_REGIONS", "us")
    odds_bookmakers: str = os.getenv("ODDS_BOOKMAKERS", "")
    train_start_season: int = int(os.getenv("MODEL_TRAIN_START_SEASON", "2023"))
    train_end_season: int = int(os.getenv("MODEL_TRAIN_END_SEASON", "2025"))
    default_min_edge: float = 0.03
    default_min_ev: float = 0.015


SETTINGS = Settings()
