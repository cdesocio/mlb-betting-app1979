from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .api_clients import fetch_odds
from .config import SETTINGS
from .data_pipeline import add_pregame_rolling_features, fetch_completed_games_for_seasons, fetch_today_prediction_frame
from .modeling import load_moneyline_model, predict_home_win_probability, train_moneyline_model
from .odds_logic import attach_betting_edges, extract_best_moneyline_prices, select_value_bets


def ensure_model_trained() -> str:
    try:
        load_moneyline_model()
        return "Loaded existing model."
    except FileNotFoundError:
        dataset = fetch_completed_games_for_seasons(SETTINGS.train_start_season, SETTINGS.train_end_season)
        dataset = add_pregame_rolling_features(dataset)
        train_result = train_moneyline_model(dataset)
        return (
            f"Trained new model with {train_result.train_rows + train_result.test_rows} rows | "
            f"log_loss={train_result.log_loss_value:.4f} | brier={train_result.brier_value:.4f} | auc={train_result.auc_value:.4f}"
        )


def build_daily_card(min_edge: float | None = None, min_ev: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    status = ensure_model_trained()
    model = load_moneyline_model()

    prediction_frame = fetch_today_prediction_frame()
    if prediction_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), status + " | No MLB games found for today."

    prediction_frame = prediction_frame.copy()
    prediction_frame["model_home_win_prob"] = predict_home_win_probability(model, prediction_frame)

    odds_payload = fetch_odds(SETTINGS.odds_api_key, SETTINGS.odds_regions, SETTINGS.odds_bookmakers)
    odds_df = extract_best_moneyline_prices(odds_payload)
    merged = attach_betting_edges(prediction_frame, odds_df)

    picks = select_value_bets(
        merged,
        min_edge=min_edge if min_edge is not None else SETTINGS.default_min_edge,
        min_ev=min_ev if min_ev is not None else SETTINGS.default_min_ev,
    )
    return picks, merged, status


if __name__ == "__main__":
    picks, merged, status = build_daily_card()
    print(status)
    if picks.empty:
        print("No bets qualified today.")
    else:
        print(picks.to_string(index=False))
