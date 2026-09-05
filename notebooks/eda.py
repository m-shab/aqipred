import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def fetch_data():
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
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

os.makedirs("eda_plots", exist_ok=True)

df = fetch_data()
print(f"Loaded {len(df)} rows")
print(df.describe())

# 1. AQI over time
plt.figure(figsize=(14, 4))
plt.plot(df["timestamp"], df["aqi"], linewidth=0.8, color="#e74c3c")
plt.title("AQI Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("eda_plots/aqi_over_time.png", dpi=120)
plt.close()

# 2. AQI distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["aqi"], bins=50, kde=True, color="#3498db")
plt.title("AQI Distribution", fontsize=14)
plt.xlabel("AQI")
plt.tight_layout()
plt.savefig("eda_plots/aqi_distribution.png", dpi=120)
plt.close()

# 3. AQI by hour
plt.figure(figsize=(12, 4))
df.groupby("hour")["aqi"].mean().plot(kind="bar", color="#2ecc71")
plt.title("Average AQI by Hour of Day", fontsize=14)
plt.xlabel("Hour")
plt.ylabel("Mean AQI")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_plots/aqi_by_hour.png", dpi=120)
plt.close()

# 4. AQI by month
plt.figure(figsize=(10, 4))
df.groupby("month")["aqi"].mean().plot(kind="bar", color="#9b59b6")
plt.title("Average AQI by Month", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Mean AQI")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_plots/aqi_by_month.png", dpi=120)
plt.close()

# 5. AQI by day of week
plt.figure(figsize=(10, 4))
df.groupby("day")["aqi"].mean().plot(kind="bar", color="#e67e22")
plt.xticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], rotation=0)
plt.title("Average AQI by Day of Week", fontsize=14)
plt.ylabel("Mean AQI")
plt.tight_layout()
plt.savefig("eda_plots/aqi_by_day.png", dpi=120)
plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(12, 8))
cols = ["aqi","pm25","pm10","no2","o3","temperature","humidity","wind_speed","precipitation"]
corr = df[cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("eda_plots/correlation_heatmap.png", dpi=120)
plt.close()

# 7. Rolling average
df["aqi_7day_avg"] = df["aqi"].rolling(24*7).mean()
plt.figure(figsize=(14, 4))
plt.plot(df["timestamp"], df["aqi"], alpha=0.3, label="Hourly AQI", color="#e74c3c")
plt.plot(df["timestamp"], df["aqi_7day_avg"], label="7-Day Rolling Avg", linewidth=2, color="#2c3e50")
plt.title("AQI with 7-Day Rolling Average", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("eda_plots/aqi_rolling_avg.png", dpi=120)
plt.close()

print("EDA plots saved to eda_plots/")