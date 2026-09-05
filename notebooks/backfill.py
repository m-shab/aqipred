import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

#lahore coordinates
LAT = 31.5497
LON = 74.3436

#backfill for 1 year prior
END_DATE   = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

def fetch_historical_weather():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,precipitation"
    )
    return requests.get(url).json()["hourly"]

def fetch_historical_aq():
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&hourly=us_aqi,pm10,pm2_5,nitrogen_dioxide,ozone"
    )
    return requests.get(url).json()["hourly"]

def build_rows(weather, aq):
    rows = []
    total = len(weather["time"])
    print(f"Building {total} rows...")
    prev_aqi = None

    for i in range(total):
        aqi = aq["us_aqi"][i]
        if aqi is None:
            prev_aqi = None
            continue

        dt = datetime.fromisoformat(weather["time"][i])
        aqi_change_rate = aqi - prev_aqi if prev_aqi is not None else 0
        prev_aqi = aqi

        rows.append({
            "timestamp"      : dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour"           : dt.hour,
            "day"            : dt.weekday(),
            "month"          : dt.month,
            "aqi"            : aqi,
            "aqi_change_rate": aqi_change_rate,
            "pm25"           : aq["pm2_5"][i],
            "pm10"           : aq["pm10"][i],
            "no2"            : aq["nitrogen_dioxide"][i],
            "o3"             : aq["ozone"][i],
            "temperature"    : weather["temperature_2m"][i],
            "humidity"       : weather["relative_humidity_2m"][i],
            "wind_speed"     : weather["windspeed_10m"][i],
            "precipitation"  : weather["precipitation"][i],
        })

    return rows

def push_to_supabase(rows):
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        supabase.table("features").insert(chunk).execute()
        print(f"Inserted rows {i} to {i+len(chunk)}")

if __name__ == "__main__":
    print("Clearing existing data...")
    supabase.table("features").delete().neq("timestamp", 0).execute()

    print(f"Backfilling from {START_DATE} to {END_DATE}...")
    weather = fetch_historical_weather()
    aq = fetch_historical_aq()
    rows = build_rows(weather, aq)
    print(f"Total rows: {len(rows)}")
    push_to_supabase(rows)
    print("Done.")