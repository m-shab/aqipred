import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FEATURES = [
    "hour", "day", "month", "aqi_change_rate",
    "pm10", "no2", "o3",
    "temperature", "humidity", "wind_speed", "precipitation"
]
TARGET = "aqi"

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
    print(f"Total rows: {len(df)}")
    return df

def prepare_data(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"{name:20s} → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
    return {"name": name, "model": model, "rmse": rmse}

def train(X_train, X_test, y_train, y_test):
    results = []
    results.append(evaluate("Ridge Regression", Ridge().fit(X_train, y_train), X_test, y_test))
    results.append(evaluate("Random Forest",    RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train), X_test, y_test))
    results.append(evaluate("XGBoost",          xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0).fit(X_train, y_train), X_test, y_test))
    best = min(results, key=lambda x: x["rmse"])
    print(f"\nBest model: {best['name']}")
    return best["model"], best["name"]

def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    with open("models/best_model_name.txt", "w") as f:
        f.write(name)
    print(f"Model saved: {name}")

if __name__ == "__main__":
    df = fetch_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    print("\nTraining models...")
    best_model, best_name = train(X_train, X_test, y_train, y_test)
    save_model(best_model, best_name)