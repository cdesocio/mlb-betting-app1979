from __future__ import annotations

from typing import Any

import pandas as pd


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def implied_prob_from_american(odds: int) -> float:
    return 1 / american_to_decimal(odds)


def remove_two_way_vig(home_odds: int, away_odds: int) -> tuple[float, float]:
    raw_home = implied_prob_from_american(home_odds)
    raw_away = implied_prob_from_american(away_odds)
    total = raw_home + raw_away
    return raw_home / total, raw_away / total


def expected_value(probability: float, american_odds: int) -> float:
    decimal_odds = american_to_decimal(american_odds)
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


def extract_best_moneyline_prices(odds_payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in odds_payload:
        home_team = event.get("home_team")
        away_team = next((t for t in event.get("teams", []) if t != home_team), None)
        best_home = None
        best_away = None
        best_home_book = ""
        best_away_book = ""

        for bookmaker in event.get("bookmakers", []):
            book_name = bookmaker.get("title", "")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = int(outcome.get("price"))
                    if name == home_team:
                        if best_home is None or american_to_decimal(price) > american_to_decimal(best_home):
                            best_home = price
                            best_home_book = book_name
                    elif name == away_team:
                        if best_away is None or american_to_decimal(price) > american_to_decimal(best_away):
                            best_away = price
                            best_away_book = book_name

        if home_team and away_team and best_home is not None and best_away is not None:
            rows.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "best_home_odds": int(best_home),
                    "best_away_odds": int(best_away),
                    "best_home_book": best_home_book,
                    "best_away_book": best_away_book,
                    "commence_time": event.get("commence_time"),
                }
            )
    return pd.DataFrame(rows)


def attach_betting_edges(predictions: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(odds_df, on=["home_team", "away_team"], how="inner")
    if merged.empty:
        return merged

    fair_probs = merged.apply(
        lambda row: remove_two_way_vig(int(row["best_home_odds"]), int(row["best_away_odds"])),
        axis=1,
        result_type="expand",
    )
    fair_probs.columns = ["market_home_prob_fair", "market_away_prob_fair"]
    merged = pd.concat([merged, fair_probs], axis=1)

    merged["model_away_win_prob"] = 1.0 - merged["model_home_win_prob"]
    merged["home_edge"] = merged["model_home_win_prob"] - merged["market_home_prob_fair"]
    merged["away_edge"] = merged["model_away_win_prob"] - merged["market_away_prob_fair"]
    merged["home_ev"] = merged.apply(
        lambda row: expected_value(float(row["model_home_win_prob"]), int(row["best_home_odds"])), axis=1
    )
    merged["away_ev"] = merged.apply(
        lambda row: expected_value(float(row["model_away_win_prob"]), int(row["best_away_odds"])), axis=1
    )
    return merged


def select_value_bets(merged: pd.DataFrame, min_edge: float, min_ev: float) -> pd.DataFrame:
    picks: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        if float(row["home_edge"]) >= min_edge and float(row["home_ev"]) >= min_ev:
            picks.append(
                {
                    "game": f"{row['away_team']} @ {row['home_team']}",
                    "bet_team": row["home_team"],
                    "opponent": row["away_team"],
                    "side": "home",
                    "odds": int(row["best_home_odds"]),
                    "book": row["best_home_book"],
                    "model_prob": float(row["model_home_win_prob"]),
                    "market_prob": float(row["market_home_prob_fair"]),
                    "edge": float(row["home_edge"]),
                    "ev": float(row["home_ev"]),
                    "starter": row.get("home_starter", ""),
                }
            )
        if float(row["away_edge"]) >= min_edge and float(row["away_ev"]) >= min_ev:
            picks.append(
                {
                    "game": f"{row['away_team']} @ {row['home_team']}",
                    "bet_team": row["away_team"],
                    "opponent": row["home_team"],
                    "side": "away",
                    "odds": int(row["best_away_odds"]),
                    "book": row["best_away_book"],
                    "model_prob": float(row["model_away_win_prob"]),
                    "market_prob": float(row["market_away_prob_fair"]),
                    "edge": float(row["away_edge"]),
                    "ev": float(row["away_ev"]),
                    "starter": row.get("away_starter", ""),
                }
            )
    out = pd.DataFrame(picks)
    if not out.empty:
        out = out.sort_values(["ev", "edge"], ascending=False).reset_index(drop=True)
    return out
