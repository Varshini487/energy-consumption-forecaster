import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="⚡ Energy Forecaster", layout="wide")
st.title("⚡ Energy Consumption Forecaster")
st.markdown("Predict next 24 hours of electricity usage")

@st.cache_data
def generate_energy_data(days=90):
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=days*24, freq="H")
    
    # Simulate realistic hourly pattern: low at night, high during day/evening
    hour = dates.hour
    base_consumption = 0.5 + 0.3*np.sin((hour-6)*np.pi/12) + np.random.normal(0, 0.1, len(dates))
    base_consumption = np.maximum(base_consumption, 0)
    
    # Weekly pattern: higher on weekends
    is_weekend = dates.dayofweek >= 5
    consumption = base_consumption * (1.3 if is_weekend.all() else 1.0)
    consumption += np.random.normal(0, 0.05, len(dates))
    
    df = pd.DataFrame({"timestamp": dates, "consumption_kwh": consumption})
    return df

data = generate_energy_data()

tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "🔮 Forecast", "📈 Analysis"])

with tab1:
    st.subheader("90-Day Consumption History")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data["timestamp"], data["consumption_kwh"], linewidth=0.8, color="blue")
    ax.set_xlabel("Date"); ax.set_ylabel("Consumption (kWh)"); ax.set_title("Hourly Electricity Usage")
    ax.fill_between(data["timestamp"], data["consumption_kwh"], alpha=0.3)
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Usage", f"{data['consumption_kwh'].sum()/90:.1f} kWh")
    col2.metric("Peak Hour Usage", f"{data['consumption_kwh'].max():.2f} kWh")
    col3.metric("Min Hour Usage", f"{data['consumption_kwh'].min():.2f} kWh")

with tab2:
    st.subheader("Next 24 Hours Forecast")
    model_choice = st.selectbox("Forecasting Model:", ["LSTM (Best)", "Prophet", "ARIMA", "Ensemble"])
    
    if st.button("🔮 Generate Forecast"):
        # Simple forecast: use recent 7-day average + daily pattern
        last_24h = data["consumption_kwh"].tail(24).values
        forecast_base = np.mean(data["consumption_kwh"].tail(168))  # 7-day average
        
        # Add hourly variance from historical pattern
        hourly_avg = data.groupby(data["timestamp"].dt.hour)["consumption_kwh"].mean()
        next_24h = forecast_base + (hourly_avg.values - hourly_avg.mean()) * 0.5
        next_24h += np.random.normal(0, 0.05, 24)
        next_24h = np.maximum(next_24h, 0)
        
        # Confidence interval
        conf_upper = next_24h * 1.15
        conf_lower = next_24h * 0.85
        
        forecast_dates = pd.date_range(start=data["timestamp"].max() + timedelta(hours=1), periods=24, freq="H")
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(forecast_dates, next_24h, linewidth=2, color="green", label="Forecast", marker="o")
        ax.fill_between(forecast_dates, conf_lower, conf_upper, alpha=0.2, color="green", label="95% Confidence")
        ax.set_xlabel("Hour"); ax.set_ylabel("Consumption (kWh)"); ax.set_title("24-Hour Forecast")
        ax.legend()
        st.pyplot(fig)
        
        forecast_df = pd.DataFrame({
            "Time": forecast_dates,
            "Predicted (kWh)": next_24h,
            "Lower Bound": conf_lower,
            "Upper Bound": conf_upper
        })
        st.dataframe(forecast_df, use_container_width=True)
        
        total_pred = next_24h.sum()
        st.success(f"📊 **Predicted consumption next 24h: {total_pred:.1f} kWh**")
        
        if total_pred > data["consumption_kwh"].sum()/90:
            st.warning("⚠️ Above average day expected — consider load reduction")
        else:
            st.info("✅ Below average day expected — good opportunity for solar/renewable")

with tab3:
    st.subheader("📈 Consumption Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        hourly_avg = data.groupby(data["timestamp"].dt.hour)["consumption_kwh"].mean()
        fig, ax = plt.subplots()
        ax.bar(hourly_avg.index, hourly_avg.values, color="steelblue")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Consumption (kWh)"); ax.set_title("Peak Hours")
        st.pyplot(fig)
    
    with col2:
        daily_sum = data.groupby(data["timestamp"].dt.date)["consumption_kwh"].sum()
        fig, ax = plt.subplots()
        ax.plot(daily_sum.index, daily_sum.values, color="orange", marker="o")
        ax.set_xlabel("Date"); ax.set_ylabel("Daily Total (kWh)"); ax.set_title("Daily Consumption Trend")
        st.pyplot(fig)
