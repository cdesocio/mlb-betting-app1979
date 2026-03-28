from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import SETTINGS
from src.run_phase1 import build_daily_card

st.set_page_config(page_title="MLB Phase 1 Betting Model", page_icon="⚾", layout="wide")

st.title("⚾ MLB Phase 1 Betting Model")
st.caption("Moneyline-only daily card with a simple calibrated logistic model and value-bet filters.")

with st.sidebar:
    st.header("Filters")
    min_edge = st.slider("Minimum edge", min_value=0.0, max_value=0.10, value=float(SETTINGS.default_min_edge), step=0.005)
    min_ev = st.slider("Minimum EV", min_value=0.0, max_value=0.10, value=float(SETTINGS.default_min_ev), step=0.005)
    run_button = st.button("Refresh today's card", type="primary")
    st.markdown("**Setup**")
    st.code("ODDS_API_KEY=...", language="bash")
    st.caption("Add your key in Streamlit secrets when deployed, or export it locally in your terminal.")

if run_button or "loaded_once" not in st.session_state:
    with st.spinner("Building today's card..."):
        try:
            picks, merged, status = build_daily_card(min_edge=min_edge, min_ev=min_ev)
            st.session_state["picks"] = picks
            st.session_state["merged"] = merged
            st.session_state["status"] = status
            st.session_state["loaded_once"] = True
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.stop()

picks = st.session_state.get("picks", pd.DataFrame())
merged = st.session_state.get("merged", pd.DataFrame())
status = st.session_state.get("status", "")

st.info(status)

col1, col2, col3 = st.columns(3)
col1.metric("Qualified bets", 0 if picks.empty else len(picks))
col2.metric("Games priced", 0 if merged.empty else len(merged))
col3.metric("Model type", "Moneyline")

st.subheader("Today's picks")
if picks.empty:
    st.write("No bets cleared your thresholds today.")
else:
    display = picks.copy()
    for column in ["model_prob", "market_prob", "edge", "ev"]:
        display[column] = (display[column] * 100).round(2).astype(str) + "%"
    st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("All priced games")
if merged.empty:
    st.write("No odds/game matches were found.")
else:
    game_view = merged[
        [
            "away_team",
            "home_team",
            "away_starter",
            "home_starter",
            "best_away_odds",
            "best_home_odds",
            "model_home_win_prob",
            "home_edge",
            "away_edge",
            "home_ev",
            "away_ev",
        ]
    ].copy()
    for column in ["model_home_win_prob", "home_edge", "away_edge", "home_ev", "away_ev"]:
        game_view[column] = (game_view[column] * 100).round(2).astype(str) + "%"
    st.dataframe(game_view, use_container_width=True, hide_index=True)

with st.expander("What Phase 1 includes"):
    st.markdown(
        """
        - Moneyline only
        - Historical MLB game results from the Stats API
        - Rolling team-form features over recent games
        - Calibrated logistic regression for home win probability
        - Best available price across books from The Odds API
        - Edge and EV filters before a bet is shown
        """
    )

with st.expander("Recommended Phase 2"):
    st.markdown(
        """
        Add totals, bullpen-fatigue features, weather, and line-movement tracking after Phase 1 is logging stable results.
        """
    )
