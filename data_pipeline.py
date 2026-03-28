from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .api_clients import fetch_schedule

TEAM_ALIASES = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


@dataclass
class RollingState:
    wins: deque[int]
    runs_for: deque[int]
    runs_against: deque[int]
    home_wins: deque[int]
    away_wins: deque[int]


WINDOW = 10
HOME_ADVANTAGE_LOGIT = 0.035


def _empty_state() -> RollingState:
    return RollingState(
        wins=deque(maxlen=WINDOW),
        runs_for=deque(maxlen=WINDOW),
        runs_against=deque(maxlen=WINDOW),
        home_wins=deque(maxlen=WINDOW),
        away_wins=deque(maxlen=WINDOW),
    )


def _team_code(full_name: str) -> str:
    return TEAM_ALIASES.get(full_name, full_name)


def _parse_game_row(game: dict[str, Any]) -> dict[str, Any]:
    teams = game["teams"]
    home = teams["home"]
    away = teams["away"]

    home_name = home["team"]["name"]
    away_name = away["team"]["name"]
    home_pitcher = home.get("probablePitcher", {}).get("fullName", "")
    away_pitcher = away.get("probablePitcher", {}).get("fullName", "")

    return {
        "game_id": str(game["gamePk"]),
        "game_datetime": pd.to_datetime(game["gameDate"], utc=True),
        "season": int(game.get("season", pd.Timestamp.utcnow().year)),
        "status": game.get("status", {}).get("detailedState", ""),
        "home_team": home_name,
        "away_team": away_name,
        "home_code": _team_code(home_name),
        "away_code": _team_code(away_name),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "home_starter": home_pitcher,
        "away_starter": away_pitcher,
    }


def schedule_to_frame(games: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [_parse_game_row(game) for game in games]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["game_datetime", "game_id"]).reset_index(drop=True)
    return df


def fetch_completed_games_for_seasons(start_season: int, end_season: int) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        start = date(season, 3, 1)
        end = date(season, 11, 30)
        df = schedule_to_frame(fetch_schedule(start, end, game_type="R"))
        if df.empty:
            continue
        finished = df[df["status"].isin(["Final", "Game Over", "Completed Early"])].copy()
        finished = finished.dropna(subset=["home_score", "away_score"])
        all_frames.append(finished)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True).sort_values(["game_datetime", "game_id"]).reset_index(drop=True)


def add_pregame_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games.copy()

    states: dict[str, RollingState] = defaultdict(_empty_state)
    rows: list[dict[str, Any]] = []

    for _, game in games.sort_values(["game_datetime", "game_id"]).iterrows():
        home = game["home_code"]
        away = game["away_code"]
        home_state = states[home]
        away_state = states[away]

        def avg_or_default(values: deque[int | float], default: float) -> float:
            return float(np.mean(values)) if values else default

        row = game.to_dict()
        row.update(
            {
                "home_recent_win_pct": avg_or_default(home_state.wins, 0.5),
                "away_recent_win_pct": avg_or_default(away_state.wins, 0.5),
                "home_recent_runs_for": avg_or_default(home_state.runs_for, 4.4),
                "away_recent_runs_for": avg_or_default(away_state.runs_for, 4.4),
                "home_recent_runs_against": avg_or_default(home_state.runs_against, 4.4),
                "away_recent_runs_against": avg_or_default(away_state.runs_against, 4.4),
                "home_home_win_pct": avg_or_default(home_state.home_wins, 0.5),
                "away_away_win_pct": avg_or_default(away_state.away_wins, 0.5),
            }
        )
        rows.append(row)

        if pd.notna(game.get("home_score")) and pd.notna(game.get("away_score")):
            home_score = int(game["home_score"])
            away_score = int(game["away_score"])
            home_win = int(home_score > away_score)
            away_win = 1 - home_win
            home_state.wins.append(home_win)
            home_state.runs_for.append(home_score)
            home_state.runs_against.append(away_score)
            home_state.home_wins.append(home_win)

            away_state.wins.append(away_win)
            away_state.runs_for.append(away_score)
            away_state.runs_against.append(home_score)
            away_state.away_wins.append(away_win)

    out = pd.DataFrame(rows)
    out["win_pct_diff"] = out["home_recent_win_pct"] - out["away_recent_win_pct"]
    out["runs_for_diff"] = out["home_recent_runs_for"] - out["away_recent_runs_for"]
    out["runs_against_diff"] = out["away_recent_runs_against"] - out["home_recent_runs_against"]
    out["split_edge"] = out["home_home_win_pct"] - out["away_away_win_pct"]
    out["home_indicator"] = 1.0
    if "home_score" in out.columns and "away_score" in out.columns:
        out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)
        out["total_runs"] = out["home_score"].astype(float) + out["away_score"].astype(float)
    return out


FEATURE_COLUMNS = [
    "home_recent_win_pct",
    "away_recent_win_pct",
    "home_recent_runs_for",
    "away_recent_runs_for",
    "home_recent_runs_against",
    "away_recent_runs_against",
    "home_home_win_pct",
    "away_away_win_pct",
    "win_pct_diff",
    "runs_for_diff",
    "runs_against_diff",
    "split_edge",
    "home_indicator",
]


def fetch_today_prediction_frame(target_date: date | None = None, lookback_days: int = 180) -> pd.DataFrame:
    target_date = target_date or datetime.now(timezone.utc).date()
    history_start = target_date - timedelta(days=lookback_days)
    history_end = target_date - timedelta(days=1)

    historical_games = schedule_to_frame(fetch_schedule(history_start, history_end, game_type="R"))
    historical_games = historical_games[
        historical_games["status"].isin(["Final", "Game Over", "Completed Early"])
    ].copy()

    today_games = schedule_to_frame(fetch_schedule(target_date, target_date, game_type="R"))
    if today_games.empty:
        return today_games

    combined = pd.concat([historical_games, today_games], ignore_index=True, sort=False)
    featured = add_pregame_rolling_features(combined)
    return featured[featured["game_datetime"].dt.date == target_date].copy()
