# 🇵🇰 Lahore AQI Predictor

A serverless, end-to-end Air Quality Index (AQI) prediction system for Lahore, Punjab. Forecasts AQI up to 3 days ahead using machine learning, with a fully automated data and training pipeline.

---

## 🔗 Live App

**[View the dashboard →](https://aqipred-gw8n8.streamlit.app/)**

---

## What It Does

- Fetches real-time weather and air quality data from Open-Meteo APIs every hour
- Engineers features (lag values, rolling averages, pollutant ratios) and stores them in Supabase
- Trains and evaluates multiple ML models daily, automatically selecting the best performer
- Displays live AQI, 3-day forecasts, SHAP explanations, EDA plots, and health alerts on an interactive Streamlit dashboard

---

## Architecture

```
Weather & Air Quality APIs (Open-Meteo)
        │
        ▼ every hour
feature_pipeline.py  ──────────────────► Supabase (Feature Store)
                                              │
                                              ▼ every day at 07:00 PKT
                                        training.py
                                              │
                                         ┌────┴────┐
                                         ▼         ▼
                                     eda_plots/  models/
                                     (7 plots)  (3 .pkl files + SHAP)
                                              │
                                              ▼
                                        app/app.py (Streamlit)
```

---

## Update Schedule

| What | How Often | When |
|---|---|---|
| Feature data (AQI, weather, pollutants) | Every hour | :00 of every hour UTC |
| ML models retrained | Every day | 02:00 UTC (07:00 PKT) |
| EDA plots regenerated | Every day | Same run as training |
| SHAP explanations updated | Every day | Same run as training |

All updates are fully automated via GitHub Actions — no manual intervention needed.

---

## Models

Five models are trained and evaluated for each forecast horizon (Day 1, 2, 3). The best model by RMSE is automatically selected and saved.

| Model | Type |
|---|---|
| Ridge Regression | Statistical baseline |
| Random Forest | Ensemble (sklearn) |
| XGBoost | Gradient boosting |
| XGBoost Tuned | Gradient boosting (tuned) |
| LightGBM | Gradient boosting |

Models are evaluated on RMSE, MAE, and R² against a persistence baseline (last known AQI). A model is only saved if it beats the baseline.

---

##  Features Used

| Feature | Description |
|---|---|
| `hour`, `day`, `month` | Time-based cyclical features |
| `aqi_lag_24/48/72` | AQI from 1, 2, 3 days ago |
| `aqi_rolling_mean_24` | 24-hour rolling mean AQI |
| `aqi_rolling_std_24` | 24-hour AQI variability |
| `aqi_change_rate` | AQI change from previous hour |
| `pm25`, `pm10`, `no2`, `o3` | Pollutant concentrations |
| `pm_ratio` | PM2.5 / PM10 ratio |
| `temperature`, `humidity` | Weather conditions |
| `wind_speed`, `precipitation` | Dispersion factors |

---

## Dashboard Pages

- **Live Forecast** — current AQI, conditions strip, 4-day timeline, 3-day outlook cards, SHAP prediction breakdown
- **Health Tips** — actionable advice based on current and forecasted AQI
- **SHAP Explanations** — full SHAP summary plots per model
- **EDA & Analysis** — 7 exploratory plots from historical data
- **Model Registry** — which model won for each forecast day
- **Feature Guide** — current live values for all 18 model features

---

## Alerts

The dashboard displays a prominent alert banner whenever AQI exceeds safe thresholds:

| Level | AQI | Action |
|---|---|---|
| Unhealthy | > 150 | Limit outdoor time, wear N95 |
| Very Unhealthy | > 200 | Avoid all outdoor activity |
| Hazardous | > 300 | Stay indoors, seal windows |

---

## Running Locally

**Prerequisites:** Python 3.11+, a Supabase project, `.env` file with credentials.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# 3. Seed historical data (run once)
python notebooks/backfill.py

# 4. Generate EDA plots
python notebooks/eda.py

# 5. Train models
python notebooks/training.py

# 6. Run the app
streamlit run app/app.py
```

After initial setup, only `feature_pipeline.py` (hourly) and `training.py` (daily) need to run — GitHub Actions handles this automatically in production.

---

## Project Structure

```
├── app/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   ├── backfill.py             # One-time historical data seed
│   ├── feature_pipeline.py     # Hourly feature ingestion
│   ├── eda.py                  # EDA plot generation
│   └── training.py             # Model training & selection
├── models/
│   ├── model_day1.pkl          # Best model for Day 1 forecast
│   ├── model_day2.pkl          # Best model for Day 2 forecast
│   ├── model_day3.pkl          # Best model for Day 3 forecast
│   ├── last_updated.txt        # Timestamp of last training run
│   └── shap/                   # SHAP JSON + PNG files
├── eda_plots/                  # Generated EDA plots
├── .github/workflows/
│   ├── feature_pipeline.yml    # Hourly GitHub Actions workflow
│   └── training.yml            # Daily GitHub Actions workflow
└── requirements.txt
```

---

## Tech Stack

- **Data:** Open-Meteo (weather + air quality APIs)
- **Feature Store:** Supabase (PostgreSQL)
- **ML:** scikit-learn, XGBoost, LightGBM, SHAP
- **Dashboard:** Streamlit
- **CI/CD:** GitHub Actions
- **Hosting:** Streamlit Community Cloud