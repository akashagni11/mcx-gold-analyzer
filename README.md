# 🥇 MCX Gold Signal System

A Streamlit-based technical analysis tool for MCX Gold trading recommendations.

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Features

### Tab 1 — Live Signal
- Real-time MCX Gold price in INR (COMEX × USD/INR)
- BUY / SELL / WAIT recommendation with score
- Plain-English pros & cons explanation

### Tab 2 — Details & Weights
- Full indicator breakdown table with current values
- Adjustable weightage sliders for each indicator
- Buy & Sell threshold configuration

### Tab 3 — Backtesting
- 1 year of 1-hour historical data
- Price chart with BUY/SELL markers
- Trade log with P&L per trade
- Win rate & total P&L summary

---

## Indicators Used
| Indicator | Default Weight |
|---|---|
| RSI (14) | 15% |
| MACD (12/26/9) | 15% |
| Moving Averages (SMA/EMA) | 15% |
| US Dollar Index (DXY) | 10% |
| Bollinger Bands | 15% |
| Stochastic (14,3) | 10% |
| VWAP | 15% |
| ATR (Volatility context) | 5% |

---

## Data Source
- Gold price: `GC=F` (COMEX Gold Futures) via `yfinance`
- USD/INR: `INR=X` via `yfinance`
- DXY: `DX-Y.NYB` via `yfinance`
- Price converted to INR per 10g (MCX unit)
- Data refreshes every 5 minutes

---

## Notes
- This tool is for educational/research purposes only
- Always use proper risk management when trading
- MCX prices may differ slightly from COMEX-derived prices due to exchange spreads and taxes
