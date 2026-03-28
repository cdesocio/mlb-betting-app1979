# MLB betting model - Phase 1

This is a working Phase 1 starter project for a daily MLB betting workflow:

- trains a moneyline model from historical MLB regular-season results
- builds pregame rolling team-form features
- pulls today's MLB slate from the MLB Stats API
- pulls moneyline odds from The Odds API
- compares model probability against de-vigged market probability
- shows only bets that clear edge and EV thresholds
- runs in Streamlit so it is easy to use on an iPhone or a MacBook browser

## Why this is the right Phase 1

Phase 1 stays intentionally narrow:

- **bet type:** moneyline only
- **delivery:** Streamlit app for browser/mobile access
- **data:** official MLB schedule/stats feed plus sportsbook odds
- **model:** calibrated logistic regression with rolling team features

That keeps the first version usable and testable before expanding to totals and bullpen/weather features in Phase 2.

## Project structure

```text
mlb_phase1/
  app.py
  requirements.txt
  .env.example
  .streamlit/config.toml
  src/
    api_clients.py
    config.py
    data_pipeline.py
    modeling.py
    odds_logic.py
    run_phase1.py
```

## Setup locally on your MacBook

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ODDS_API_KEY="YOUR_KEY_HERE"
streamlit run app.py
```

Then open the local URL in your browser.

## Deploy so it works well on iPhone + MacBook

The cleanest option is **Streamlit Community Cloud**.

1. Put this project in a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Deploy the repo and point it at `app.py`.
4. In the app settings, add `ODDS_API_KEY` under **Secrets**.
5. Open the public/private app URL on your iPhone 17 Pro Max and bookmark it to your home screen.

## Required environment variables

- `ODDS_API_KEY` - required
- `ODDS_REGIONS` - optional, default `us`
- `ODDS_BOOKMAKERS` - optional comma-separated list if you only want specific books
- `MODEL_TRAIN_START_SEASON` - optional, default `2023`
- `MODEL_TRAIN_END_SEASON` - optional, default `2025`

## Notes about the model

The model uses only team-level rolling performance features in Phase 1:

- recent win rate
- recent runs scored
- recent runs allowed
- home/away split form

This is enough to create a disciplined baseline. It is **not** enough to claim a durable betting edge by itself. You should track every recommendation and compare it to the closing line before trusting bankroll to it.

## What Phase 2 should add

After you have a few weeks of logged Phase 1 results, Phase 2 should add:

- totals model
- bullpen usage/fatigue
- weather
- line movement tracking
- reporting on closing-line value and ROI by edge bucket

## Suggested operating schedule

- **Daily:** run the app once in the morning and once 30-60 minutes before first pitch
- **Weekly:** review hit rate, ROI, and closing-line value
- **After 3-4 weeks:** move to Phase 2 only if Phase 1 data logging is stable

## Important limitation

This project helps identify **candidate positive-EV bets**. It does not guarantee profit.
