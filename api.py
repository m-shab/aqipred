# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # --- Test AQICN ---
# aqicn_token = os.getenv("AQICN_TOKEN")
# url = f"https://api.waqi.info/feed/lahore/?token={aqicn_token}"
# data = requests.get(url).json()

# print("=== AQICN ===")
# print("Status :", data["status"])
# print("AQI    :", data["data"]["aqi"])
# print("PM2.5  :", data["data"]["iaqi"].get("pm25", {}).get("v", "N/A"))
# print("PM10   :", data["data"]["iaqi"].get("pm10", {}).get("v", "N/A"))
# print()

# # --- Test Open-Meteo Weather ---
# weather_url = (
#     "https://api.open-meteo.com/v1/forecast"
#     "?latitude=31.5497&longitude=74.3436"
#     "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
#     "&forecast_days=1"
# )
# w = requests.get(weather_url).json()

# print("=== Open-Meteo Weather ===")
# print("Temp     :", w["hourly"]["temperature_2m"][0], "°C")
# print("Humidity :", w["hourly"]["relativehumidity_2m"][0], "%")
# print("Wind     :", w["hourly"]["windspeed_10m"][0], "km/h")
# print()

# # --- Test Open-Meteo Air Quality ---
# aq_url = (
#     "https://air-quality-api.open-meteo.com/v1/air-quality"
#     "?latitude=31.5497&longitude=74.3436"
#     "&hourly=pm10,pm2_5,nitrogen_dioxide,ozone"
#     "&forecast_days=3"
# )
# aq = requests.get(aq_url).json()

# print("=== Open-Meteo Air Quality ===")
# print("PM2.5    :", aq["hourly"]["pm2_5"][0])
# print("PM10     :", aq["hourly"]["pm10"][0])
# print("Rows     :", len(aq["hourly"]["pm2_5"]), "(should be 72)")
import requests

def pm25_to_aqi(pm25):
    # US EPA breakpoints
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for low_pm, high_pm, low_aqi, high_aqi in breakpoints:
        if low_pm <= pm25 <= high_pm:
            aqi = ((high_aqi - low_aqi) / (high_pm - low_pm)) * (pm25 - low_pm) + low_aqi
            return round(aqi)
    return 500

aq_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=31.5497&longitude=74.3436"
    "&hourly=pm2_5&forecast_days=1"
)
aq = requests.get(aq_url).json()

from datetime import datetime
current_hour = datetime.now().hour
pm25 = aq["hourly"]["pm2_5"][current_hour]
aqi = pm25_to_aqi(pm25)

print(f"PM2.5 : {pm25}")
print(f"AQI   : {aqi}")