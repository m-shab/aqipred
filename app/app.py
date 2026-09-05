import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIG & CONSTANTS ---
LAT = 31.5497
LON = 74.3436

FEATURES = [
    "hour", "day", "month", "aqi_change_rate",
    "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
    "aqi_rolling_mean_24", "aqi_rolling_std_24",
    "pm25", "pm10", "no2", "o3", "pm_ratio",
    "temperature", "humidity", "wind_speed", "precipitation"
]

FEATURE_DESCRIPTIONS = {
    "hour":                "Hour of the day (0–23)",
    "day":                 "Day of the week (0=Mon, 6=Sun)",
    "month":               "Month of the year (1–12)",
    "aqi_change_rate":     "AQI change from previous hour",
    "aqi_lag_24":          "AQI 24 hours ago",
    "aqi_lag_48":          "AQI 48 hours ago",
    "aqi_lag_72":          "AQI 72 hours ago",
    "aqi_rolling_mean_24": "Mean AQI over past 24 hours",
    "aqi_rolling_std_24":  "AQI variability over past 24 hours",
    "pm25":                "Fine particulate matter ≤2.5 µm",
    "pm10":                "Coarse particulate matter ≤10 µm",
    "no2":                 "Nitrogen dioxide µg/m³",
    "o3":                  "Ozone µg/m³",
    "pm_ratio":            "PM2.5 to PM10 ratio",
    "temperature":         "Air temperature °C",
    "humidity":            "Relative humidity %",
    "wind_speed":          "Wind speed km/h",
    "precipitation":       "Rainfall mm",
}

# --- PALETTE ---
C_DARK    = "#2d4847"   
C_SAGE    = "#6b8f71"   
C_STEEL   = "#5b8fa8"   
C_LIGHT   = "#87c4cf"   
C_CREAM   = "#eee8c4"   
C_CREAM2  = "#e4dcb4"   
C_INK     = "#1c2e2c"   
C_MUTED   = "#7a9a94"   

PKT = pytz.timezone("Asia/Karachi")

# --- DATA FETCHING ---
def fetch_forecast():
    try:
        w = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
            "&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,precipitation&forecast_days=1"
        ).json()["hourly"]
        a = requests.get(
            f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}"
            "&hourly=us_aqi,pm10,pm2_5,nitrogen_dioxide,ozone&forecast_days=4"
        ).json()["hourly"]
        return w, a
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None, None

def build_features(w, a):
    if not w or not a: return None
    now = datetime.now(PKT)
    h   = now.hour
    vals        = [v for v in a["us_aqi"] if v is not None]
    cur         = a["us_aqi"][h] or 0
    prev        = a["us_aqi"][h-1] if h > 0 else cur
    pm25        = a["pm2_5"][h] or 0
    pm10        = a["pm10"][h] or 0
    recent      = vals[-24:] if len(vals) >= 24 else vals
    return {
        "hour": h, "day": now.weekday(), "month": now.month,
        "aqi_change_rate":     cur - (prev or cur),
        "aqi_lag_24":          vals[-24] if len(vals)>=24 else cur,
        "aqi_lag_48":          vals[-48] if len(vals)>=48 else cur,
        "aqi_lag_72":          vals[-72] if len(vals)>=72 else cur,
        "aqi_rolling_mean_24": float(np.mean(recent)),
        "aqi_rolling_std_24":  float(np.std(recent)),
        "pm25": pm25, "pm10": pm10,
        "no2":  a["nitrogen_dioxide"][h] or 0,
        "o3":   a["ozone"][h] or 0,
        "pm_ratio": pm25/(pm10+1e-6),
        "temperature":   w["temperature_2m"][h],
        "humidity":      w["relative_humidity_2m"][h],
        "wind_speed":    w["windspeed_10m"][h],
        "precipitation": w["precipitation"][h],
        "current_aqi":   cur,
    }

def aqi_meta(v):
    if v <=  50: return "Good",                           C_SAGE,  "#d6ede0"
    if v <= 100: return "Moderate",                       "#b8860b","#f5edce"
    if v <= 150: return "Sensitive Groups",               C_STEEL, "#d6e8f0"
    if v <= 200: return "Unhealthy",                      "#b34040","#f0d6d6"
    if v <= 300: return "Very Unhealthy",                 "#7a4a8a","#e8d6f0"
    return             "Hazardous",                       "#8a2020","#f0d0d0"

def tips_for(aqi):
    if aqi <= 50:
        return [("✅", "Great day outdoors", "Air quality is excellent. Enjoy outdoor activities freely."), ("🏃", "Exercise outside", "Perfect conditions for running."), ("🪟", "Open your windows", "Let fresh air circulate.")]
    if aqi <= 100:
        return [("😷", "Sensitive groups", "People with respiratory issues should monitor symptoms."), ("🏃", "Outdoor exercise", "Most can exercise normally."), ("🪟", "Ventilation", "Windows can remain open.")]
    if aqi <= 150:
        return [("😷", "Mask recommended", "N95/KN95 helps reduce particle inhalation."), ("🏠", "Limit outdoor time", "Keep exposure under 30 minutes."), ("🌿", "Use air purifiers", "Run HEPA filters indoors.")]
    if aqi <= 200:
        return [("🚫", "Avoid exertion", "Reduce outdoor activity to a minimum."), ("🏠", "Keep windows closed", "Seal gaps and run air purifiers."), ("💊", "Check medications", "Keep inhalers accessible.")]
    return [("🚨", "Stay indoors", "Serious health risk. Avoid going outside."), ("🏠", "Seal your home", "Use damp towels for air gaps."), ("📞", "Monitor health", "Seek help if breathing is difficult.")]

def model_name(day):
    p = f"models/model_day{day}_name.txt"
    return open(p).read().strip() if os.path.exists(p) else "XGBoost Regressor"

@st.cache_resource
def load_models():
    try:
        return (joblib.load("models/model_day1.pkl"), joblib.load("models/model_day2.pkl"), joblib.load("models/model_day3.pkl"))
    except:
        return None, None, None

# --- UI SETUP ---
st.set_page_config(page_title="Lahore AQI", page_icon="🇵🇰", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, .stApp {{
    font-family: 'DM Sans', sans-serif !important;
    background: {C_CREAM} !important;
    color: {C_INK} !important;
}}

/* Fix content alignment */
.block-container {{ 
    padding: 2rem 3rem !important; 
    max-width: 1200px !important;
}}

/* Keep Header for Sidebar Toggle but style it */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* Sidebar toggle button — dark background so the white icon is visible */
[data-testid="stSidebarCollapsedControl"] button,
button[data-testid="stSidebarNavToggleButton"] {{
    background-color: {C_DARK} !important;
    border-radius: 8px !important;
}}

#MainMenu, footer {{ visibility: hidden; }}

section[data-testid="stSidebar"] > div:first-child {{
    background: {C_DARK};
}}

/* Sidebar Styling */
.sb-logo {{ padding: 1.5rem; border-bottom: 1px solid rgba(238,232,196,0.1); margin-bottom: 1rem; }}
.sb-logo h2 {{ font-size: 1.1rem; color: {C_CREAM}; margin:0; }}
.sb-logo p {{ font-size: 0.7rem; color: {C_MUTED}; margin-top: 0.2rem; }}

/* Top Bar */
.top-bar {{ margin-bottom: 2rem; }}
.top-bar h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem; }}
.top-bar-sub {{ font-size: 0.85rem; color: {C_MUTED}; }}

/* AQI Hero */
.aqi-hero {{
    display: flex;
    flex-wrap: wrap;
    gap: 3rem;
    padding: 2rem;
    background: {C_CREAM2};
    border-radius: 20px;
    align-items: center;
    margin-bottom: 2rem;
}}
.aqi-number {{ font-size: 6rem; font-weight: 800; line-height: 1; letter-spacing: -0.04em; }}
.aqi-label {{ font-size: 1.4rem; font-weight: 600; }}

/* Scale */
.scale-track {{
    height: 10px; width: 100%; border-radius: 10px;
    background: linear-gradient(to right, {C_SAGE}, #b8860b, {C_STEEL}, #b34040, #7a4a8a, #8a2020);
    position: relative; margin: 1rem 0;
}}
.scale-pin {{
    position: absolute; top: 50%; transform: translate(-50%, -50%);
    width: 18px; height: 18px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}}

/* Condition Strip */
.cond-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 1px; background: {C_CREAM2};
    border-radius: 15px; overflow: hidden; margin-bottom: 2.5rem;
}}
.cond-cell {{ background: {C_CREAM}; padding: 1.2rem 0.5rem; text-align: center; }}
.cond-val {{ font-size: 1.1rem; font-weight: 700; }}
.cond-lbl {{ font-size: 0.65rem; color: {C_MUTED}; text-transform: uppercase; }}

/* Cards */
.fc-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }}
.fc-card {{
    background: {C_CREAM2}; padding: 1.5rem; border-radius: 15px; text-align: center;
}}
.fc-num {{ font-size: 3rem; font-weight: 700; }}

.mcard {{
    background: {C_CREAM2}; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;
}}
.mcard-info {{ flex: 1; min-width: 200px; }}

/* Table */
.ftbl {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
.ftbl th {{ text-align: left; padding: 0.8rem; border-bottom: 2px solid {C_CREAM2}; color: {C_MUTED}; font-size: 0.7rem; text-transform: uppercase; }}
.ftbl td {{ padding: 0.8rem; border-bottom: 1px solid {C_CREAM2}; font-size: 0.85rem; }}
.fname {{ font-family: 'DM Mono', monospace; font-weight: 600; color: {C_DARK}; }}

/* Buttons in Sidebar */
.stButton>button {{
    background: transparent !important;
    color: rgba(238,232,196,0.7) !important;
    border: none !important;
    text-align: left !important;
    padding: 0.5rem 1rem !important;
    transition: 0.2s;
}}
.stButton>button:hover {{
    color: {C_LIGHT} !important;
    background: rgba(135,196,207,0.1) !important;
}}
</style>
""", unsafe_allow_html=True)

# --- DATA PROCESSING ---
with st.spinner("Fetching data..."):
    weather, aq = fetch_forecast()
    feat = build_features(weather, aq)

if feat:
    m1, m2, m3 = load_models()
    X = pd.DataFrame([{k: v for k, v in feat.items() if k != "current_aqi"}])[FEATURES]
    
    # Placeholder if models don't exist
    d1 = max(0, round(m1.predict(X)[0])) if m1 else 145
    d2 = max(0, round(m2.predict(X)[0])) if m2 else 152
    d3 = max(0, round(m3.predict(X)[0])) if m3 else 138

    today = datetime.now(PKT).date()
    fc = [
        (today + timedelta(days=1), d1, model_name(1)),
        (today + timedelta(days=2), d2, model_name(2)),
        (today + timedelta(days=3), d3, model_name(3)),
    ]
    cur = feat["current_aqi"]
    c_lbl, c_col, c_bg = aqi_meta(cur)
    change = int(feat["aqi_change_rate"])
    arrow = "↑" if change > 0 else "↓" if change < 0 else "—"

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f'<div class="sb-logo"><h2>🇵🇰 Lahore AQI</h2><p>Machine Learning Forecast<br>{datetime.now(PKT).strftime("%H:%M PKT")}</p></div>', unsafe_allow_html=True)
        
        pages = ["Live Forecast", "Health Tips", "SHAP Explanations", "EDA & Analysis", "Model Registry", "Feature Guide"]
        if "page" not in st.session_state: st.session_state.page = "Live Forecast"
        
        for pg in pages:
            if st.button(pg, use_container_width=True):
                st.session_state.page = pg
                st.rerun()

    # --- MAIN CONTENT ---
    page = st.session_state.page
    st.markdown(f'<div class="top-bar"><h1>{page}</h1><div class="top-bar-sub">Lahore, Punjab · {datetime.now(PKT).strftime("%A, %d %B")}</div></div>', unsafe_allow_html=True)

    if page == "Live Forecast":
        pct = min(cur / 300 * 100, 100)
        st.markdown(f"""
        <div class="aqi-hero">
            <div style="flex: 0 0 200px">
                <div class="aqi-number" style="color:{c_col}">{cur}</div>
                <div class="aqi-label" style="color:{c_col}">{c_lbl}</div>
                <div style="font-size:0.8rem; color:{C_MUTED}">{arrow} {abs(change)} from last hour</div>
            </div>
            <div style="flex: 1; min-width: 300px;">
                <div style="font-size:0.7rem; font-weight:700; color:{C_MUTED}; letter-spacing:0.1em">AQI SCALE</div>
                <div class="scale-track"><div class="scale-pin" style="left:{pct}%; background:{c_col}"></div></div>
                <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:{C_MUTED}">
                    <span>0 GOOD</span><span>100 MODERATE</span><span>200 UNHEALTHY</span><span>300+</span>
                </div>
                <p style="margin-top:1.5rem; font-size:0.9rem; line-height:1.6; color:{C_INK}">
                    The current air quality in Lahore is <b>{c_lbl.lower()}</b>. 
                    {tips_for(cur)[0][2]}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cond-strip">
            <div class="cond-cell"><div class="cond-lbl">Temp</div><div class="cond-val">{feat['temperature']:.0f}°C</div></div>
            <div class="cond-cell"><div class="cond-lbl">Humidity</div><div class="cond-val">{feat['humidity']:.0f}%</div></div>
            <div class="cond-cell"><div class="cond-lbl">Wind</div><div class="cond-val">{feat['wind_speed']:.0f} km/h</div></div>
            <div class="cond-cell"><div class="cond-lbl">PM2.5</div><div class="cond-val">{feat['pm25']:.0f} <span style="font-size:0.6rem;font-weight:400">µg/m³</span></div></div>
            <div class="cond-cell"><div class="cond-lbl">PM10</div><div class="cond-val">{feat['pm10']:.0f} <span style="font-size:0.6rem;font-weight:400">µg/m³</span></div></div>
            <div class="cond-cell"><div class="cond-lbl">NO₂</div><div class="cond-val">{feat['no2']:.0f} <span style="font-size:0.6rem;font-weight:400">µg/m³</span></div></div>
            <div class="cond-cell"><div class="cond-lbl">O₃</div><div class="cond-val">{feat['o3']:.0f} <span style="font-size:0.6rem;font-weight:400">µg/m³</span></div></div>
        </div>
        """, unsafe_allow_html=True)

        # --- TIMELINE CHART ---
        st.subheader("4-Day AQI Timeline")
        timeline_labels = ["Today"] + [date.strftime("%a\n%d %b") for date, _, _ in fc]
        timeline_values = [cur, d1, d2, d3]
        timeline_colors = [aqi_meta(v)[1] for v in timeline_values]

        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_facecolor("#e4dcb4")
        ax.set_facecolor("#e4dcb4")

        # shaded AQI bands
        bands = [(0,50,"#d6ede0"), (50,100,"#f5edce"), (100,150,"#d6e8f0"),
                 (150,200,"#f0d6d6"), (200,300,"#e8d6f0")]
        for lo, hi, bc in bands:
            ax.axhspan(lo, hi, color=bc, alpha=0.35, zorder=0)

        xs = list(range(4))
        ax.plot(xs, timeline_values, color="#2d4847", linewidth=2.5,
                zorder=2, solid_capstyle="round")
        for x, y, c in zip(xs, timeline_values, timeline_colors):
            ax.scatter(x, y, color=c, s=120, zorder=3, edgecolors="#2d4847", linewidths=1.5)
            ax.text(x, y + 6, str(y), ha="center", va="bottom",
                    fontsize=11, fontweight="700", color="#1c2e2c")

        ax.set_xticks(xs)
        ax.set_xticklabels(timeline_labels, fontsize=10, color="#1c2e2c")
        ax.set_yticks([0, 50, 100, 150, 200, 300])
        ax.tick_params(axis="y", labelcolor="#7a9a94", labelsize=8)
        ax.set_ylim(0, max(timeline_values) + 50)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.yaxis.grid(True, color="#c8c0a0", linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close(fig)

        # --- 3-DAY FORECAST CARDS ---
        st.subheader("3-Day Outlook")
        fc_cols = st.columns(3)
        for i, (date, aqi, mname) in enumerate(fc):
            lbl, col, _ = aqi_meta(aqi)
            with fc_cols[i]:
                st.markdown(f"""
                <div class="fc-card">
                    <div style="font-size:0.8rem; font-weight:600; color:{C_MUTED}">{date.strftime('%A')}</div>
                    <div class="fc-num" style="color:{col}">{aqi}</div>
                    <div style="font-size:0.9rem; font-weight:600; color:{col}">{lbl}</div>
                    <div style="font-size:0.6rem; color:{C_MUTED}; margin-top:1rem; border-top:1px solid rgba(0,0,0,0.05); padding-top:0.5rem">{mname}</div>
                </div>
                """, unsafe_allow_html=True)

        # --- INLINE SHAP BREAKDOWN ---
        st.subheader("Prediction Breakdown")
        st.caption("Top features driving each day's forecast — green bars push AQI down, red bars push it up.")

        shap_available = any(
            os.path.exists(f"models/shap/shap_day{d}.png") for d in [1,2,3]
        )

        if shap_available:
            shap_cols = st.columns(3)
            for day_num, (date, aqi, mname) in enumerate(fc, 1):
                shap_path = f"models/shap/shap_day{day_num}.png"
                lbl, col, _ = aqi_meta(aqi)
                with shap_cols[day_num - 1]:
                    st.markdown(f"""
                    <div style="background:{C_CREAM2}; border-radius:12px; padding:1rem; margin-bottom:0.5rem;">
                        <div style="font-size:0.7rem; font-weight:700; color:{C_MUTED}; text-transform:uppercase; letter-spacing:0.08em">{date.strftime('%A, %d %b')}</div>
                        <div style="font-size:1.8rem; font-weight:800; color:{col}; line-height:1.1">{aqi}</div>
                        <div style="font-size:0.75rem; color:{col}; font-weight:600; margin-bottom:0.5rem">{lbl}</div>
                        <div style="font-size:0.6rem; color:{C_MUTED}">{mname}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(shap_path):
                        st.image(shap_path, use_container_width=True)
                    else:
                        st.caption("SHAP plot not yet generated.")
        else:
            st.info("SHAP plots will appear here after the first training run completes.")

    elif page == "Health Tips":
        for day_offset, (date, aqi, _) in enumerate([(today, cur, "")] + fc):
            lbl, col, bg = aqi_meta(aqi)
            title = "Right Now" if day_offset == 0 else date.strftime("%A, %d %b")
            st.markdown(f"### {title} ({aqi} - {lbl})")
            for icon, head, body in tips_for(aqi):
                st.info(f"**{icon} {head}**: {body}")

    elif page == "Feature Guide":
        st.write("Current calculated feature values used for the prediction:")
        rows = ""
        for f in FEATURES:
            val = feat.get(f, 0)
            desc = FEATURE_DESCRIPTIONS.get(f, "")
            rows += f"<tr><td><span class='fname'>{f}</span></td><td>{desc}</td><td><b>{val:.2f}</b></td></tr>"
        
        st.markdown(f"""
        <table class="ftbl">
            <thead><tr><th>Feature Key</th><th>Description</th><th>Current Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

    elif page == "Model Registry":
        st.write("Predictive models are updated daily based on the lowest RMSE.")
        for day_num, (date, aqi, mname) in enumerate(fc, 1):
            lbl, col, _ = aqi_meta(aqi)
            st.markdown(f"""
            <div class="mcard">
                <div class="mcard-info">
                    <div style="font-weight:700; font-size:1.1rem">Day {day_num} Forecast Model</div>
                    <div style="color:{C_STEEL}; font-size:0.9rem; margin-bottom:0.3rem">{mname}</div>
                    <div style="color:{C_MUTED}; font-size:0.8rem">Optimized for {date.strftime('%Y-%m-%d')}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:2rem; font-weight:800; color:{col}">{aqi}</div>
                    <div style="font-size:0.7rem; font-weight:700; color:{col}">{lbl.upper()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "SHAP Explanations":
        st.write("SHAP (SHapley Additive exPlanations) shows which features most influenced each day's model prediction.")
        for day_num, (date, aqi, mname) in enumerate(fc, 1):
            shap_path = f"models/shap/shap_day{day_num}.png"
            st.markdown(f"#### Day {day_num} — {date.strftime('%A, %d %b')} · {mname}")
            if os.path.exists(shap_path):
                st.image(shap_path, use_container_width=True)
            else:
                st.warning(f"SHAP plot not found at `{shap_path}`. Run the training pipeline to generate it.")
            st.markdown("---")

    elif page == "EDA & Analysis":
        st.write("Exploratory analysis of historical Lahore AQI data used to train the models.")

        eda_plots = [
            ("aqi_over_time.png",      "AQI Over Time",              "Full time-series of recorded AQI values."),
            ("aqi_distribution.png",   "AQI Distribution",           "Histogram showing how AQI values are spread."),
            ("aqi_rolling_avg.png",    "Rolling Average",            "7-day rolling mean to highlight longer-term trends."),
            ("aqi_by_hour.png",        "Average AQI by Hour",        "Intra-day pattern — peak pollution hours."),
            ("aqi_by_day.png",         "Average AQI by Day of Week", "Weekly pattern of air quality."),
            ("aqi_by_month.png",       "Average AQI by Month",       "Seasonal variation across the year."),
            ("correlation_heatmap.png","Feature Correlation Heatmap","Pearson correlations between all model features."),
        ]

        for fname, title, caption in eda_plots:
            path = f"eda_plots/{fname}"
            st.markdown(f"#### {title}")
            st.caption(caption)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Plot not found at `{path}`. Run `notebooks/eda.py` to generate EDA plots.")
            st.markdown("---")

else:
    st.error("Waiting for API response...")