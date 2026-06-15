# ⚡ Energy Consumption Forecaster

A **time series forecasting model** that predicts household electricity usage. Uses historical hourly consumption data to forecast next day's usage, helping optimize energy costs and manage grid load.

## 🔮 What It Predicts
- Next 24 hours: hourly electricity consumption (kWh)
- Confidence intervals (95%)
- Peak hours and usage trends

## 📊 Models Compared
| Model | RMSE | MAE | Best For |
|-------|------|-----|----------|
| ARIMA | 0.98 | 0.75 | Trend capture |
| Prophet | 0.82 | 0.61 | Seasonality |
| LSTM | 0.64 | 0.48 | **Complex patterns** |
| Ensemble | **0.59** | **0.44** | Production |

## 🛠️ Tech Stack
- **LSTM / GRU** – sequence-to-sequence forecasting
- **ARIMA / Prophet** – traditional time series
- **Statsmodels / Scikit-learn** – baseline models
- **Pandas** – time series manipulation
- **Streamlit** – interactive dashboard

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/energy-consumption-forecaster
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Smart grid optimization
- Home energy management
- Utility demand planning
- Cost reduction for households
