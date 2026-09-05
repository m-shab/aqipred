import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LAT = 31.5497
LON = 74.3436

def fetch_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,precipitation"
        "&forecast_days=1"
    )
    hourly = requests.get(url).json()["hourly"]
    current_hour = datetime.now().hour
    return {
        "temperature"  : hourly["temperature_2m"][current_hour],
        "humidity"     : hourly["relative_humidity_2m"][current_hour],
        "wind_speed"   : hourly["windspeed_10m"][current_hour],
        "precipitation": hourly["precipitation"][current_hour],
    }

def fetch_air_quality():
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=us_aqi,pm10,pm2_5,nitrogen_dioxide,ozone"
        "&forecast_days=1"
    )
    hourly = requests.get(url).json()["hourly"]
    current_hour = datetime.now().hour
    return {
        "aqi" : hourly["us_aqi"][current_hour],
        "pm25": hourly["pm2_5"][current_hour],
        "pm10": hourly["pm10"][current_hour],
        "no2" : hourly["nitrogen_dioxide"][current_hour],
        "o3"  : hourly["ozone"][current_hour],
    }

def fetch_previous_aqi():
    response = supabase.table("features").select("aqi").order("created_at", desc=True).limit(1).execute()
    if len(response.data) == 0:
        return None
    return response.data[0]["aqi"]

def row_exists_for_hour(ts):
    """Check if we already have a row for this exact hour to avoid duplicates."""
    response = supabase.table("features").select("timestamp").eq("timestamp", ts).execute()
    return len(response.data) > 0

def compute_features(weather, aq, prev_aqi):
    now = datetime.now()
    # Truncate to the current hour so timestamps always align with backfill data
    now_hour = now.replace(minute=0, second=0, microsecond=0)
    aqi_change_rate = aq["aqi"] - prev_aqi if prev_aqi is not None else 0
    return {
        "timestamp"      : now_hour.strftime("%Y-%m-%d %H:%M:%S"),
        "hour"           : now_hour.hour,
        "day"            : now_hour.weekday(),
        "month"          : now_hour.month,
        "aqi"            : aq["aqi"],
        "aqi_change_rate": aqi_change_rate,
        "pm25"           : aq["pm25"],
        "pm10"           : aq["pm10"],
        "no2"            : aq["no2"],
        "o3"             : aq["o3"],
        **weather,
    }

def push_to_supabase(row):
    response = supabase.table("features").insert(row).execute()
    print("Pushed to Supabase:", response.data)

if __name__ == "__main__":
    print("Fetching weather...")
    weather = fetch_weather()
    print(weather)

    print("\nFetching air quality...")
    aq = fetch_air_quality()
    print(aq)

    print("\nFetching previous AQI...")
    prev_aqi = fetch_previous_aqi()
    print("Previous AQI:", prev_aqi)

    print("\nComputing features...")
    row = compute_features(weather, aq, prev_aqi)
    print(row)

    # Skip if this hour already exists (prevents duplicate rows on re-runs)
    if row_exists_for_hour(row["timestamp"]):
        print(f"\n⚠️  Row for {row['timestamp']} already exists — skipping insert.")
    else:
        print("\nPushing to Supabase...")
        push_to_supabase(row)