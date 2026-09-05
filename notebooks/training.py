import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FEATURES = [
    "hour", "day", "month",
    "aqi_change_rate",
    "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
    "aqi_rolling_mean_24", "aqi_rolling_std_24",
    "pm25", "pm10", "no2", "o3", "pm_ratio",
    "temperature", "humidity", "wind_speed", "precipitation"
]

def fetch_data():
    print("Fetching data from Supabase...")
    all_rows = []
    page = 0
    while True:
        response = (
            supabase.table("features")
            .select("*")
            .range(page * 1000, (page + 1) * 1000 - 1)
            .execute()
        )
        if not response.data:
            break
        all_rows.extend(response.data)
        page += 1
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.now() - timedelta(days=365)
    df = df[df["timestamp"] >= cutoff]
    print(f"Total rows (1yr): {len(df)}")
    return df

def prepare_data(df, hours_ahead=24):
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_lag_24"]          = df["aqi"].shift(24)
    df["aqi_lag_48"]          = df["aqi"].shift(48)
    df["aqi_lag_72"]          = df["aqi"].shift(72)
    df["aqi_rolling_mean_24"] = df["aqi"].shift(1).rolling(24).mean()
    df["aqi_rolling_std_24"]  = df["aqi"].shift(1).rolling(24).std()
    df["pm_ratio"]            = df["pm25"] / (df["pm10"] + 1e-6)
    df["target_aqi"]          = df["aqi"].shift(-hours_ahead)

    df = df.dropna(subset=FEATURES + ["target_aqi"])
    X = df[FEATURES]
    y = df["target_aqi"]
    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"{name:25s} → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
    return {"name": name, "model": model, "rmse": rmse}

def train(X_train, X_test, y_train, y_test):
    # persistence baseline
    persistence_preds = np.full(len(y_test), y_train.iloc[-1])
    p_rmse = np.sqrt(mean_squared_error(y_test, persistence_preds))
    p_mae  = mean_absolute_error(y_test, persistence_preds)
    p_r2   = r2_score(y_test, persistence_preds)
    print(f"{'Persistence Baseline':25s} → RMSE: {p_rmse:.2f} | MAE: {p_mae:.2f} | R²: {p_r2:.3f}")
    print("-" * 60)

    results = []
    results.append(evaluate("Ridge Regression",
        Ridge().fit(X_train, y_train), X_test, y_test))

    results.append(evaluate("Random Forest",
        RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train), X_test, y_test))

    results.append(evaluate("XGBoost",
        xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0).fit(X_train, y_train), X_test, y_test))

    results.append(evaluate("XGBoost Tuned",
        xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5,
                         subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                         random_state=42, verbosity=0).fit(X_train, y_train), X_test, y_test))

    results.append(evaluate("LightGBM",
        lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                          random_state=42, verbose=-1).fit(X_train, y_train), X_test, y_test))

    best = min(results, key=lambda x: x["rmse"])
    print(f"\nBest model: {best['name']}")

    if best["rmse"] < p_rmse:
        print(f"✅ Beats persistence baseline (RMSE {best['rmse']:.2f} < {p_rmse:.2f})")
    else:
        print(f"❌ Does NOT beat persistence baseline (RMSE {best['rmse']:.2f} >= {p_rmse:.2f})")

    return best["model"], best["name"]

def save_shap(model, X_train, day, name):
    print(f"Computing SHAP for Day {day}...")
    os.makedirs("models/shap", exist_ok=True)
    try:
        sample = X_train.sample(min(200, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(f"models/shap/shap_day{day}.png", dpi=100)
        plt.close()
        print(f"SHAP plot saved for Day {day}")
    except Exception as e:
        print(f"SHAP failed for {name}: {e}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = fetch_data()

    for day, hours in [(1, 24), (2, 48), (3, 72)]:
        print(f"\n{'='*60}")
        print(f"Training model for Day {day} ({hours}h ahead)")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = prepare_data(df, hours_ahead=hours)
        best_model, best_name = train(X_train, X_test, y_train, y_test)

        joblib.dump(best_model, f"models/model_day{day}.pkl")
        with open(f"models/model_day{day}_name.txt", "w") as f:
            f.write(best_name)
        print(f"Saved model_day{day}.pkl ({best_name})")

        save_shap(best_model, X_train, day, best_name)