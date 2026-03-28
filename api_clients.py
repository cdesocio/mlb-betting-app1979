from __future__ import annotations

from datetime import date
from typing import Any

import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


class ApiError(RuntimeError):
    pass


def _get_json(url: str, params: dict[str, Any]) -> Any:
    response = requests.get(url, params=params, timeout=30)
    if response.status_code >= 400:
        raise ApiError(f"API request failed ({response.status_code}): {response.text[:200]}")
    return response.json()


def fetch_schedule(start_date: date, end_date: date, game_type: str = "R") -> list[dict[str, Any]]:
    payload = _get_json(
        MLB_SCHEDULE_URL,
        {
            "sportId": 1,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "gameType": game_type,
            "hydrate": "probablePitcher(note),team",
        },
    )
    games: list[dict[str, Any]] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            games.append(game)
    return games


def fetch_standings(season: int) -> list[dict[str, Any]]:
    payload = _get_json(
        MLB_STANDINGS_URL,
        {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"},
    )
    rows: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        rows.extend(record.get("teamRecords", []))
    return rows


def fetch_odds(api_key: str, regions: str = "us", bookmakers: str = "") -> list[dict[str, Any]]:
    if not api_key:
        raise ApiError("Missing ODDS_API_KEY. Put it in your environment or Streamlit secrets.")

    params: dict[str, Any] = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get_json(ODDS_API_URL, params)
