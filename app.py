import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MCX Gold Signal",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    .signal-buy {
        background: linear-gradient(135deg, #0d3d1f, #1a6b35);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-bottom: 20px;
    }
    .signal-sell {
        background: linear-gradient(135deg, #3d0d0d, #6b1a1a);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-bottom: 20px;
    }
    .signal-wait {
        background: linear-gradient(135deg, #3d3200, #6b5700);
        border: 2px solid #f39c12;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-bottom: 20px;
    }
    .price-box {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .pros-box {
        background: #0d2b1a;
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .cons-box {
        background: #2b0d0d;
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 16px;
    }
    .score-bar-outer {
        background: #222;
        border-radius: 10px;
        height: 22px;
        width: 100%;
        margin: 6px 0 16px 0;
    }
    h1, h2, h3 { color: #f0d080 !important; }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-size: 16px; }
    .stTabs [aria-selected="true"] { color: #f0d080 !important; border-bottom: 2px solid #f0d080; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stSlider > div > div { color: #f0d080; }
</style>
""", unsafe_allow_html=True)


# ── Data Fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_gold_data(period="1y", interval="1h"):
    gold    = yf.download("GC=F",      period=period, interval=interval, progress=False, auto_adjust=True)
    usd_inr = yf.download("INR=X",     period=period, interval=interval, progress=False, auto_adjust=True)
    dxy     = yf.download("DX-Y.NYB",  period=period, interval=interval, progress=False, auto_adjust=True)

    if gold.empty:
        return None, None, None

    # Flatten multi-level columns if present (yfinance >= 0.2.40)
    def flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()

    gold    = flatten(gold)
    usd_inr = flatten(usd_inr)
    dxy     = flatten(dxy)

    usd_inr_aligned = usd_inr['Close'].reindex(gold.index, method='ffill')
    dxy_aligned     = dxy['Close'].reindex(gold.index, method='ffill')

    # Convert to INR (troy oz → 10g MCX unit; 1 troy oz = 31.1035g → 10g = /3.11035)
    gold_inr = gold.copy()
    gold_inr['Close_INR'] = gold['Close'] * usd_inr_aligned / 3.11035
    gold_inr['Open_INR']  = gold['Open']  * usd_inr_aligned / 3.11035
    gold_inr['High_INR']  = gold['High']  * usd_inr_aligned / 3.11035
    gold_inr['Low_INR']   = gold['Low']   * usd_inr_aligned / 3.11035
    gold_inr['DXY']       = dxy_aligned
    gold_inr['USDINR']    = usd_inr_aligned

    return gold_inr, usd_inr_aligned, dxy_aligned


# ── Technical Indicators ──────────────────────────────────────────────────────
def compute_indicators(df):
    close = df['Close_INR']
    high  = df['High_INR']
    low   = df['Low_INR']
    vol   = df['Volume']

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # Moving Averages
    df['SMA20']  = close.rolling(20).mean()
    df['SMA50']  = close.rolling(50).mean()
    df['EMA9']   = close.ewm(span=9, adjust=False).mean()
    df['EMA21']  = close.ewm(span=21, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid']   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pct']   = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # VWAP (rolling daily approx for multi-day data)
    typical = (high + low + close) / 3
    df['VWAP'] = (typical * vol).rolling(20).sum() / vol.rolling(20).sum()

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    return df.dropna()


# ── Scoring Engine ────────────────────────────────────────────────────────────
def compute_score(row, weights):
    signals = {}
    reasons_bull = []
    reasons_bear = []

    close = row['Close_INR']

    # RSI
    rsi = row['RSI']
    if rsi < 30:
        s = 1; reasons_bull.append(f"RSI is deeply oversold ({rsi:.1f}) — strong bounce potential")
    elif rsi < 45:
        s = 0.5; reasons_bull.append(f"RSI is in lower range ({rsi:.1f}) — mild bullish lean")
    elif rsi > 70:
        s = -1; reasons_bear.append(f"RSI is overbought ({rsi:.1f}) — buying may be overdone")
    elif rsi > 55:
        s = -0.5; reasons_bear.append(f"RSI is elevated ({rsi:.1f}) — momentum slowing")
    else:
        s = 0
    signals['RSI'] = s

    # MACD
    macd_hist = row['MACD_Hist']
    macd      = row['MACD']
    if macd > 0 and macd_hist > 0:
        s = 1; reasons_bull.append("MACD is positive and rising — buyers are in control")
    elif macd > 0 and macd_hist < 0:
        s = 0.3; reasons_bull.append("MACD is positive but momentum is fading — caution needed")
    elif macd < 0 and macd_hist < 0:
        s = -1; reasons_bear.append("MACD is negative and falling — sellers are dominating")
    elif macd < 0 and macd_hist > 0:
        s = -0.3; reasons_bear.append("MACD is negative but recovering — wait for confirmation")
    else:
        s = 0
    signals['MACD'] = s

    # Moving Averages
    ma_score = 0
    ma_count = 0
    if close > row['SMA20']: ma_score += 1; ma_count += 1
    else: ma_score -= 1; ma_count += 1
    if close > row['SMA50']: ma_score += 1; ma_count += 1
    else: ma_score -= 1; ma_count += 1
    if close > row['EMA9']:  ma_score += 1; ma_count += 1
    else: ma_score -= 1; ma_count += 1
    if row['EMA9'] > row['EMA21']: ma_score += 1; ma_count += 1
    else: ma_score -= 1; ma_count += 1
    s = ma_score / ma_count
    signals['MA'] = s
    if s > 0.3:
        reasons_bull.append("Gold is trading above key moving averages — uptrend intact")
    elif s < -0.3:
        reasons_bear.append("Gold is below key moving averages — downtrend in play")

    # DXY (inverse relationship)
    dxy = row['DXY']
    dxy_series = row.get('DXY_prev', dxy)
    dxy_change = (dxy - dxy_series) / dxy_series * 100 if dxy_series else 0
    if dxy_change < -0.3:
        s = 1; reasons_bull.append(f"US Dollar is weakening today — typically bullish for gold")
    elif dxy_change < 0:
        s = 0.4; reasons_bull.append("Dollar is slightly down — mild tailwind for gold")
    elif dxy_change > 0.3:
        s = -1; reasons_bear.append(f"US Dollar is strengthening — headwind for gold prices")
    elif dxy_change > 0:
        s = -0.4; reasons_bear.append("Dollar ticking up slightly — watch for gold resistance")
    else:
        s = 0
    signals['DXY'] = s

    # Bollinger Bands
    bb_pct = row['BB_Pct']
    if bb_pct < 0.15:
        s = 1; reasons_bull.append("Price is near the lower Bollinger Band — potential bounce zone")
    elif bb_pct < 0.35:
        s = 0.5; reasons_bull.append("Price is in the lower half of Bollinger Bands — mild bullish")
    elif bb_pct > 0.85:
        s = -1; reasons_bear.append("Price is at the upper Bollinger Band — may face resistance here")
    elif bb_pct > 0.65:
        s = -0.5; reasons_bear.append("Price is in the upper half of Bollinger Bands — some caution")
    else:
        s = 0
    signals['BB'] = s

    # Stochastic
    k = row['Stoch_K']
    d = row['Stoch_D']
    if k < 20 and k > d:
        s = 1; reasons_bull.append("Stochastic is oversold and turning up — reversal signal")
    elif k < 30:
        s = 0.5; reasons_bull.append("Stochastic is in oversold territory — downside may be limited")
    elif k > 80 and k < d:
        s = -1; reasons_bear.append("Stochastic is overbought and turning down — sell signal")
    elif k > 70:
        s = -0.5; reasons_bear.append("Stochastic is overbought — short-term pullback possible")
    else:
        s = 0
    signals['Stoch'] = s

    # VWAP
    vwap = row['VWAP']
    diff_pct = (close - vwap) / vwap * 100
    if diff_pct > 0.3:
        s = 1; reasons_bull.append("Price is trading above VWAP — institutions are net buyers")
    elif diff_pct > 0:
        s = 0.4; reasons_bull.append("Price is just above VWAP — slight buying advantage")
    elif diff_pct < -0.3:
        s = -1; reasons_bear.append("Price is below VWAP — institutional selling pressure present")
    elif diff_pct < 0:
        s = -0.4; reasons_bear.append("Price is just below VWAP — slight selling pressure")
    else:
        s = 0
    signals['VWAP'] = s

    # ATR (used for context only — contributes a small volatility signal)
    atr = row['ATR']
    atr_pct = atr / close * 100
    if atr_pct > 0.8:
        s = 0; reasons_bear.append(f"Volatility is high (ATR {atr_pct:.2f}%) — wider stop-losses needed, trade carefully")
    else:
        s = 0
    signals['ATR'] = s

    # Weighted score → normalize to 0–100
    total_weight = sum(weights.values())
    raw_score = sum(signals.get(k, 0) * weights[k] for k in weights)
    max_possible = sum(weights.values())
    score_100 = ((raw_score / max_possible) + 1) / 2 * 100  # map -1..1 → 0..100

    return round(score_100, 1), signals, reasons_bull, reasons_bear


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df, weights, buy_threshold, sell_threshold):
    results = []
    for i in range(len(df)):
        row = df.iloc[i].copy()
        if i > 0:
            row['DXY_prev'] = df.iloc[i-1]['DXY']
        score, _, _, _ = compute_score(row, weights)
        if score >= buy_threshold:
            sig = 'BUY'
        elif score <= sell_threshold:
            sig = 'SELL'
        else:
            sig = 'WAIT'
        results.append({'Date': df.index[i], 'Close_INR': row['Close_INR'],
                        'Score': score, 'Signal': sig})

    res = pd.DataFrame(results).set_index('Date')

    # Simple P&L: enter on BUY, exit on SELL or next BUY reversal
    trades = []
    in_trade = False
    entry_price = 0
    entry_date = None

    for i in range(len(res)):
        sig = res.iloc[i]['Signal']
        price = res.iloc[i]['Close_INR']
        date  = res.index[i]
        if not in_trade and sig == 'BUY':
            in_trade = True
            entry_price = price
            entry_date = date
        elif in_trade and sig == 'SELL':
            pnl = price - entry_price
            pnl_pct = pnl / entry_price * 100
            trades.append({'Entry': entry_date, 'Exit': date,
                           'Entry_Price': entry_price, 'Exit_Price': price,
                           'PnL_INR': round(pnl, 2), 'PnL_Pct': round(pnl_pct, 2)})
            in_trade = False

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return res, trades_df


# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 🥇 MCX Gold Signal System")
st.caption("Powered by COMEX Gold × USD/INR | 1-Hour Candles | Technical Analysis")

# Default weights
DEFAULT_WEIGHTS = {'RSI': 15, 'MACD': 15, 'MA': 15, 'DXY': 10,
                   'BB': 15, 'Stoch': 10, 'VWAP': 15, 'ATR': 5}

if 'weights' not in st.session_state:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
if 'buy_thresh' not in st.session_state:
    st.session_state.buy_thresh = 62
if 'sell_thresh' not in st.session_state:
    st.session_state.sell_thresh = 38

tab1, tab2, tab3 = st.tabs(["📊 Live Signal", "⚙️ Details & Weights", "🔁 Backtesting"])

# ── Fetch Data ────────────────────────────────────────────────────────────────
with st.spinner("Fetching latest gold data..."):
    df_raw, _, _ = fetch_gold_data(period="1y", interval="1h")

if df_raw is None or df_raw.empty:
    st.error("⚠️ Could not fetch data. Please check your internet connection.")
    st.stop()

df = compute_indicators(df_raw.copy())

if df.empty:
    st.error("Not enough data to compute indicators.")
    st.stop()

latest = df.iloc[-1].copy()
prev   = df.iloc[-2]
latest['DXY_prev'] = prev['DXY']

score, signals, pros, cons = compute_score(latest, st.session_state.weights)

buy_thresh  = st.session_state.buy_thresh
sell_thresh = st.session_state.sell_thresh

if score >= buy_thresh:
    signal_label = "BUY"
    signal_class = "signal-buy"
    signal_emoji = "🟢"
    signal_color = "#2ecc71"
elif score <= sell_thresh:
    signal_label = "SELL"
    signal_class = "signal-sell"
    signal_emoji = "🔴"
    signal_color = "#e74c3c"
else:
    signal_label = "WAIT"
    signal_class = "signal-wait"
    signal_emoji = "🟡"
    signal_color = "#f39c12"

current_price = latest['Close_INR']
prev_price    = prev['Close_INR']
price_change  = current_price - prev_price
price_chg_pct = price_change / prev_price * 100


# ════════════════════ TAB 1 — LIVE SIGNAL ════════════════════════════════════
with tab1:
    # Price row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:13px;color:#aaa;'>MCX Gold (10g)</div>
            <div style='font-size:28px;font-weight:700;color:#f0d080;'>₹{current_price:,.0f}</div>
            <div style='font-size:14px;color:{"#2ecc71" if price_change>=0 else "#e74c3c"}'>
                {"▲" if price_change>=0 else "▼"} ₹{abs(price_change):,.0f} ({price_chg_pct:+.2f}%)
            </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:13px;color:#aaa;'>USD/INR</div>
            <div style='font-size:24px;font-weight:600;color:#f0f0f0;'>₹{latest['USDINR']:.2f}</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:13px;color:#aaa;'>DXY (Dollar Index)</div>
            <div style='font-size:24px;font-weight:600;color:#f0f0f0;'>{latest['DXY']:.2f}</div>
            </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:13px;color:#aaa;'>ATR (Volatility)</div>
            <div style='font-size:24px;font-weight:600;color:#f0f0f0;'>₹{latest['ATR']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main signal card
    col_sig, col_score = st.columns([1, 1])
    with col_sig:
        st.markdown(f"""<div class='{signal_class}'>
            <div style='font-size:18px;color:#ddd;margin-bottom:6px;'>Recommendation</div>
            <div style='font-size:56px;font-weight:900;color:{signal_color};letter-spacing:4px;'>{signal_emoji} {signal_label}</div>
            <div style='font-size:13px;color:#bbb;margin-top:6px;'>Thresholds: Buy ≥ {buy_thresh} | Sell ≤ {sell_thresh}</div>
        </div>""", unsafe_allow_html=True)

    with col_score:
        st.markdown(f"""<div class='{signal_class}' style='height:100%;'>
            <div style='font-size:18px;color:#ddd;margin-bottom:6px;'>Overall Score</div>
            <div style='font-size:56px;font-weight:900;color:{signal_color};'>{score}<span style='font-size:24px;color:#aaa;'>/100</span></div>
            <div style='font-size:13px;color:#bbb;margin-top:6px;'>{"Bullish" if score > 55 else "Bearish" if score < 45 else "Neutral"} bias</div>
        </div>""", unsafe_allow_html=True)

    # Score bar
    bar_color = signal_color
    st.markdown(f"""
    <div class='score-bar-outer'>
        <div style='width:{score}%;background:{bar_color};height:100%;border-radius:10px;
                    transition:width 0.5s;display:flex;align-items:center;justify-content:center;
                    font-size:12px;font-weight:700;color:#000;'>{score:.0f}</div>
    </div>
    <div style='display:flex;justify-content:space-between;font-size:11px;color:#888;margin-top:-10px;'>
        <span>0 — Strong Sell</span><span>50 — Neutral</span><span>100 — Strong Buy</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pros & Cons
    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown("### ✅ Why this looks bullish")
        if pros:
            pros_html = "".join(f"<p style='margin:6px 0;font-size:14px;'>✅ {p}</p>" for p in pros)
            st.markdown(f"<div class='pros-box'>{pros_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pros-box'><p style='color:#888;'>No strong bullish signals right now.</p></div>", unsafe_allow_html=True)

    with col_c:
        st.markdown("### ⚠️ Reasons to be cautious")
        if cons:
            cons_html = "".join(f"<p style='margin:6px 0;font-size:14px;'>⚠️ {c}</p>" for c in cons)
            st.markdown(f"<div class='cons-box'>{cons_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='cons-box'><p style='color:#888;'>No significant bearish signals right now.</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Last updated: {df.index[-1].strftime('%d %b %Y, %I:%M %p')} UTC | Data refreshes every 5 minutes")
    if st.button("🔄 Refresh Data", key="refresh1"):
        st.cache_data.clear()
        st.rerun()


# ════════════════════ TAB 2 — DETAILS & WEIGHTS ═══════════════════════════════
with tab2:
    st.markdown("### 📋 Indicator Breakdown")

    indicator_names = {
        'RSI': 'RSI (14)', 'MACD': 'MACD (12/26/9)', 'MA': 'Moving Averages',
        'DXY': 'US Dollar Index', 'BB': 'Bollinger Bands',
        'Stoch': 'Stochastic (14,3)', 'VWAP': 'VWAP', 'ATR': 'ATR (Volatility)'
    }
    indicator_values = {
        'RSI':   f"{latest['RSI']:.1f}",
        'MACD':  f"{latest['MACD']:.1f} / Hist: {latest['MACD_Hist']:.1f}",
        'MA':    f"SMA20: ₹{latest['SMA20']:,.0f} | EMA9: ₹{latest['EMA9']:,.0f}",
        'DXY':   f"{latest['DXY']:.2f}",
        'BB':    f"BB%: {latest['BB_Pct']*100:.1f}%",
        'Stoch': f"K: {latest['Stoch_K']:.1f} | D: {latest['Stoch_D']:.1f}",
        'VWAP':  f"₹{latest['VWAP']:,.0f}",
        'ATR':   f"₹{latest['ATR']:,.0f}",
    }

    rows = []
    for k, name in indicator_names.items():
        s = signals.get(k, 0)
        if s > 0.3:   sig_str, sig_col = "Bullish ✅", "#2ecc71"
        elif s < -0.3: sig_str, sig_col = "Bearish 🔴", "#e74c3c"
        else:          sig_str, sig_col = "Neutral ➖", "#f39c12"
        contribution = round(s * st.session_state.weights[k] / 2 + st.session_state.weights[k] / 2, 1)
        rows.append({'Indicator': name, 'Current Value': indicator_values[k],
                     'Signal': sig_str, 'Weight': f"{st.session_state.weights[k]}%",
                     'Score Contrib': f"{contribution:.1f}/{st.session_state.weights[k]}"})

    breakdown_df = pd.DataFrame(rows)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Signal": st.column_config.TextColumn("Signal"),
                 })

    st.markdown("---")
    st.markdown("### ⚖️ Indicator Weightages")
    st.info("Adjust the weight of each indicator. Total should sum to 100%.")

    w = st.session_state.weights
    col1, col2 = st.columns(2)
    with col1:
        w['RSI']   = st.slider("RSI",              0, 30, w['RSI'],   step=1)
        w['MACD']  = st.slider("MACD",             0, 30, w['MACD'],  step=1)
        w['MA']    = st.slider("Moving Averages",  0, 30, w['MA'],    step=1)
        w['DXY']   = st.slider("US Dollar Index",  0, 20, w['DXY'],   step=1)
    with col2:
        w['BB']    = st.slider("Bollinger Bands",  0, 30, w['BB'],    step=1)
        w['Stoch'] = st.slider("Stochastic",       0, 20, w['Stoch'], step=1)
        w['VWAP']  = st.slider("VWAP",             0, 30, w['VWAP'],  step=1)
        w['ATR']   = st.slider("ATR (Volatility)", 0, 15, w['ATR'],   step=1)

    total_w = sum(w.values())
    if total_w != 100:
        st.warning(f"⚠️ Total weight = {total_w}%. Please adjust to exactly 100%.")
    else:
        st.success("✅ Weights sum to 100%")

    st.markdown("---")
    st.markdown("### 🎯 Signal Thresholds")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.session_state.buy_thresh  = st.slider("Buy Threshold (score ≥)",  50, 90, st.session_state.buy_thresh,  step=1)
    with tc2:
        st.session_state.sell_thresh = st.slider("Sell Threshold (score ≤)", 10, 50, st.session_state.sell_thresh, step=1)

    if st.button("✅ Apply & Recalculate", key="apply"):
        st.session_state.weights = w
        st.rerun()


# ════════════════════ TAB 3 — BACKTESTING ════════════════════════════════════
with tab3:
    st.markdown("### 🔁 Backtest — 1 Year of 1H Candles")
    st.caption("Simulates BUY/SELL signals on historical data using your current weights & thresholds.")

    if st.button("▶️ Run Backtest", key="run_bt"):
        with st.spinner("Running backtest on 1 year of data..."):
            bt_results, trades_df = run_backtest(df, st.session_state.weights,
                                                  st.session_state.buy_thresh,
                                                  st.session_state.sell_thresh)

        # ── Price chart with signals
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            subplot_titles=("MCX Gold Price (INR) with Signals", "Score", "RSI"))

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open_INR'], high=df['High_INR'],
            low=df['Low_INR'], close=df['Close_INR'],
            name="Gold", increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
        ), row=1, col=1)

        buy_dates  = bt_results[bt_results['Signal'] == 'BUY'].index
        sell_dates = bt_results[bt_results['Signal'] == 'SELL'].index

        fig.add_trace(go.Scatter(x=buy_dates,  y=df.loc[buy_dates, 'Low_INR']  * 0.999,
                                  mode='markers', marker=dict(symbol='triangle-up', size=10, color='#2ecc71'),
                                  name='BUY'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_dates, y=df.loc[sell_dates, 'High_INR'] * 1.001,
                                  mode='markers', marker=dict(symbol='triangle-down', size=10, color='#e74c3c'),
                                  name='SELL'), row=1, col=1)

        fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Score'],
                                  line=dict(color='#f0d080', width=1), name='Score'), row=2, col=1)
        fig.add_hline(y=st.session_state.buy_thresh,  line_dash="dot", line_color="#2ecc71", row=2, col=1)
        fig.add_hline(y=st.session_state.sell_thresh, line_dash="dot", line_color="#e74c3c", row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                  line=dict(color='#9b59b6', width=1), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#e74c3c", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#2ecc71", row=3, col=1)

        fig.update_layout(
            template='plotly_dark', height=700,
            paper_bgcolor='#0f0f0f', plot_bgcolor='#0f0f0f',
            xaxis_rangeslider_visible=False, showlegend=True,
            legend=dict(bgcolor='#1a1a2e', bordercolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary stats
        st.markdown("### 📈 Performance Summary")
        total_signals = len(bt_results[bt_results['Signal'] != 'WAIT'])
        buy_count  = len(buy_dates)
        sell_count = len(sell_dates)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total BUY Signals",  buy_count)
        m2.metric("Total SELL Signals", sell_count)

        if not trades_df.empty:
            wins     = trades_df[trades_df['PnL_INR'] > 0]
            losses   = trades_df[trades_df['PnL_INR'] <= 0]
            win_rate = len(wins) / len(trades_df) * 100
            total_pnl = trades_df['PnL_INR'].sum()

            m3.metric("Win Rate",   f"{win_rate:.1f}%")
            m4.metric("Total P&L",  f"₹{total_pnl:,.0f}", delta=f"{'▲' if total_pnl>0 else '▼'} {abs(total_pnl):,.0f}")

            st.markdown("### 🧾 Trade Log")
            trades_display = trades_df.copy()
            trades_display['Entry_Price'] = trades_display['Entry_Price'].apply(lambda x: f"₹{x:,.0f}")
            trades_display['Exit_Price']  = trades_display['Exit_Price'].apply(lambda x: f"₹{x:,.0f}")
            trades_display['PnL_INR']     = trades_display['PnL_INR'].apply(lambda x: f"₹{x:,.0f}")
            trades_display['PnL_Pct']     = trades_display['PnL_Pct'].apply(lambda x: f"{x:+.2f}%")
            trades_display.columns = ['Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 'P&L (₹)', 'P&L (%)']
            st.dataframe(trades_display, use_container_width=True, hide_index=True)

            st.markdown("### 💡 Threshold Suggestion")
            st.info(f"""Based on this backtest:
- **Win Rate:** {win_rate:.1f}% | **Total Trades:** {len(trades_df)}
- Current thresholds → Buy ≥ **{st.session_state.buy_thresh}** | Sell ≤ **{st.session_state.sell_thresh}**
- Try adjusting thresholds in Tab 2 and re-running to find the optimal setup.""")
        else:
            m3.metric("Completed Trades", "0")
            m4.metric("Total P&L", "₹0")
            st.warning("No completed BUY→SELL trade pairs found. Try lowering the Buy threshold or raising the Sell threshold.")