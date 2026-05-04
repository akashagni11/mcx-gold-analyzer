import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MCX Gold Advisor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    .signal-buy {
        background: linear-gradient(135deg, #0d3d1f, #1a6b35);
        border: 2px solid #2ecc71; border-radius: 16px;
        padding: 28px; text-align: center; margin-bottom: 16px;
    }
    .signal-sell {
        background: linear-gradient(135deg, #3d0d0d, #6b1a1a);
        border: 2px solid #e74c3c; border-radius: 16px;
        padding: 28px; text-align: center; margin-bottom: 16px;
    }
    .signal-wait {
        background: linear-gradient(135deg, #3d3200, #6b5700);
        border: 2px solid #f39c12; border-radius: 16px;
        padding: 28px; text-align: center; margin-bottom: 16px;
    }
    .price-box {
        background: #1a1a2e; border: 1px solid #333;
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .score-block {
        background: #1a1a2e; border: 1px solid #333;
        border-radius: 12px; padding: 16px; text-align: center;
        margin-bottom: 10px;
    }
    .pros-box {
        background: #0d2b1a; border-left: 4px solid #2ecc71;
        border-radius: 8px; padding: 16px; margin-bottom: 12px;
    }
    .cons-box {
        background: #2b0d0d; border-left: 4px solid #e74c3c;
        border-radius: 8px; padding: 16px;
    }
    .bar-outer {
        background: #222; border-radius: 10px; height: 20px;
        width: 100%; margin: 4px 0 14px 0;
    }
    .manual-box {
        background: #1a1a2e; border: 1px solid #444;
        border-radius: 10px; padding: 16px; margin-top: 10px;
    }
    h1, h2, h3 { color: #f0d080 !important; }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-size: 15px; }
    .stTabs [aria-selected="true"] { color: #f0d080 !important; border-bottom: 2px solid #f0d080; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def fetch_all_data(period="5y"):
    def dl(ticker):
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()

    gold   = dl("GC=F")
    usd_inr = dl("INR=X")
    dxy    = dl("DX-Y.NYB")
    tnx    = dl("^TNX")
    gld    = dl("GLD")
    silver = dl("SI=F")

    if gold.empty:
        return None

    idx = gold.index

    def align(s, col='Close'):
        return s[col].reindex(idx, method='ffill')

    df = gold.copy()
    df['USDINR'] = align(usd_inr)
    df['DXY']    = align(dxy)
    df['TNX']    = align(tnx)
    df['GLD']    = align(gld)
    df['SILVER'] = align(silver)

    df['Close_INR'] = df['Close'] * df['USDINR'] / 3.11035
    df['Open_INR']  = df['Open']  * df['USDINR'] / 3.11035
    df['High_INR']  = df['High']  * df['USDINR'] / 3.11035
    df['Low_INR']   = df['Low']   * df['USDINR'] / 3.11035

    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def compute_indicators(df):
    close = df['Close_INR']
    high  = df['High_INR']
    low   = df['Low_INR']

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # Moving Averages
    df['SMA20']  = close.rolling(20).mean()
    df['SMA50']  = close.rolling(50).mean()
    df['SMA200'] = close.rolling(200).mean()
    df['EMA9']   = close.ewm(span=9,  adjust=False).mean()
    df['EMA21']  = close.ewm(span=21, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid']   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    df['BB_Pct']   = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR']     = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / close * 100

    # ADX
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm]   = 0
    minus_dm[minus_dm < plus_dm]  = 0
    atr14    = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX']      = dx.rolling(14).mean()
    df['Plus_DI']  = plus_di
    df['Minus_DI'] = minus_di

    # Gold/Silver ratio
    df['GS_Ratio'] = df['Close'] / (df['SILVER'] + 1e-9)

    return df.dropna()


def add_rolling_features(df):
    df = df.copy()
    df['TNX_30d_chg'] = df['TNX'].diff(30)
    df['GLD_20d_chg'] = df['GLD'].pct_change(20) * 100
    df['GLD_50d_chg'] = df['GLD'].pct_change(50) * 100
    df['GS_90d_avg']  = df['GS_Ratio'].rolling(90).mean()
    df['DXY_30d_chg'] = df['DXY'].diff(30)
    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL SCORING
# ══════════════════════════════════════════════════════════════════════════════
def score_technical(row, prev_row, tech_weights):
    signals = {}
    bull, bear = [], []
    close = row['Close_INR']

    # RSI
    rsi = row['RSI']
    if rsi < 30:
        s = 1.0;  bull.append(f"RSI is deeply oversold ({rsi:.0f}) — historically a strong buying zone for gold")
    elif rsi < 42:
        s = 0.5;  bull.append(f"RSI is in lower range ({rsi:.0f}) — mild bullish lean")
    elif rsi > 72:
        s = -1.0; bear.append(f"RSI is overbought ({rsi:.0f}) — gold may be overextended, wait for a dip")
    elif rsi > 58:
        s = -0.5; bear.append(f"RSI is elevated ({rsi:.0f}) — momentum slowing, not the ideal entry")
    else:
        s = 0.0
    signals['RSI'] = s

    # MACD
    hist      = row['MACD_Hist']
    macd      = row['MACD']
    prev_hist = prev_row['MACD_Hist'] if prev_row is not None else hist
    if macd > 0 and hist > 0 and hist > prev_hist:
        s = 1.0;  bull.append("MACD is rising and positive — buyers are firmly in control")
    elif macd > 0 and hist > 0:
        s = 0.5;  bull.append("MACD is positive — upward momentum, though slowing slightly")
    elif macd < 0 and hist < 0 and hist < prev_hist:
        s = -1.0; bear.append("MACD is falling and negative — sellers are dominating right now")
    elif macd < 0 and hist < 0:
        s = -0.5; bear.append("MACD is negative — downward pressure, caution on buying now")
    elif macd < 0 and hist > 0:
        s = 0.2;  bull.append("MACD is recovering from negative — early sign of a potential reversal")
    else:
        s = 0.0
    signals['MACD'] = s

    # Moving Averages
    ma_score     = 0
    above_sma50  = close > row['SMA50']
    above_sma200 = close > row['SMA200']
    above_sma20  = close > row['SMA20']
    golden_cross = row['SMA50'] > row['SMA200']

    if above_sma200: ma_score += 1
    else: ma_score -= 1
    if above_sma50:  ma_score += 1
    else: ma_score -= 1
    if above_sma20:  ma_score += 0.5
    else: ma_score -= 0.5
    if golden_cross: ma_score += 1
    else: ma_score -= 1

    s = max(-1, min(1, ma_score / 3.5))
    signals['MA'] = s
    if s > 0.4:
        bull.append("Gold is trading above its key long-term averages — the broader uptrend is intact")
    elif s < -0.4:
        bear.append("Gold is below key moving averages — the trend is currently against buyers")
    if golden_cross:
        bull.append("The 50-day average is above the 200-day (Golden Cross) — a classic long-term bullish signal")
    else:
        bear.append("The 50-day average is below the 200-day (Death Cross) — long-term trend is bearish")

    # ADX
    adx      = row['ADX']
    plus_di  = row['Plus_DI']
    minus_di = row['Minus_DI']
    if adx > 25 and plus_di > minus_di:
        s = 1.0;  bull.append(f"ADX is strong ({adx:.0f}) with buyers leading — the uptrend has real conviction")
    elif adx > 25 and minus_di > plus_di:
        s = -1.0; bear.append(f"ADX is strong ({adx:.0f}) with sellers leading — avoid buying into this downtrend")
    elif adx > 20 and plus_di > minus_di:
        s = 0.5;  bull.append(f"ADX shows a developing uptrend ({adx:.0f}) — momentum building in gold's favour")
    elif adx < 20:
        s = 0.0;  bear.append(f"ADX is weak ({adx:.0f}) — gold is choppy with no clear trend, wait for direction")
    else:
        s = -0.3
    signals['ADX'] = s

    # Bollinger Bands
    bb = row['BB_Pct']
    if bb < 0.1:
        s = 1.0;  bull.append("Price is at the lower Bollinger Band — gold is statistically cheap in the current range, good entry zone")
    elif bb < 0.35:
        s = 0.5;  bull.append("Price is in the lower half of Bollinger Bands — mild buying opportunity")
    elif bb > 0.9:
        s = -1.0; bear.append("Price is at the upper Bollinger Band — gold is statistically expensive here, not ideal to buy now")
    elif bb > 0.65:
        s = -0.5; bear.append("Price is in the upper half of Bollinger Bands — wait for a pullback for a better price")
    else:
        s = 0.0
    signals['BB'] = s

    # Stochastic
    k = row['Stoch_K']
    d = row['Stoch_D']
    if k < 20 and k > d:
        s = 1.0;  bull.append(f"Stochastic is oversold and turning up ({k:.0f}) — short-term reversal signal")
    elif k < 25:
        s = 0.5;  bull.append(f"Stochastic in oversold zone ({k:.0f}) — downside may be limited from here")
    elif k > 80 and k < d:
        s = -1.0; bear.append(f"Stochastic is overbought and turning down ({k:.0f}) — pullback likely")
    elif k > 75:
        s = -0.5; bear.append(f"Stochastic overbought ({k:.0f}) — not the best time to buy, wait for cooling off")
    else:
        s = 0.0
    signals['Stoch'] = s

    # ATR
    atr_pct = row['ATR_Pct']
    if atr_pct > 1.5:
        bear.append(f"Volatility is very high (ATR {atr_pct:.1f}%) — prices swinging wildly, consider waiting for stability")
        s = -0.3
    else:
        s = 0.0
    signals['ATR'] = s

    total_w   = sum(tech_weights.values())
    raw       = sum(signals.get(k, 0) * tech_weights[k] for k in tech_weights)
    score_100 = ((raw / total_w) + 1) / 2 * 100
    return round(score_100, 1), signals, bull, bear


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL SCORING
# ══════════════════════════════════════════════════════════════════════════════
def score_fundamental(row, prev_row, fund_weights):
    signals = {}
    bull, bear = [], []

    # US 10-Year Treasury Yield
    tnx     = row['TNX']
    tnx_30d = row.get('TNX_30d_chg', 0)
    if pd.isna(tnx_30d): tnx_30d = 0

    if tnx < 3.5 and tnx_30d < -0.2:
        s = 1.0;  bull.append(f"US 10-yr yield is low and falling ({tnx:.2f}%) — very bullish for gold as bonds become less attractive")
    elif tnx < 4.0 and tnx_30d < 0:
        s = 0.6;  bull.append(f"US bond yields are declining ({tnx:.2f}%) — supports gold as a store of value")
    elif tnx > 5.0 and tnx_30d > 0.2:
        s = -1.0; bear.append(f"US 10-yr yield is high and rising ({tnx:.2f}%) — strong competition from bonds, bearish for gold")
    elif tnx > 4.5 and tnx_30d > 0:
        s = -0.6; bear.append(f"Bond yields are elevated ({tnx:.2f}%) — reduces appeal of holding gold")
    elif tnx_30d < -0.1:
        s = 0.4;  bull.append(f"Bond yields have been declining recently ({tnx:.2f}%) — mild tailwind for gold")
    elif tnx_30d > 0.1:
        s = -0.4; bear.append(f"Bond yields have been rising ({tnx:.2f}%) — mild headwind for gold")
    else:
        s = 0.0
    signals['TNX'] = s

    # GLD ETF Trend
    gld_20d = row.get('GLD_20d_chg', 0)
    gld_50d = row.get('GLD_50d_chg', 0)
    if pd.isna(gld_20d): gld_20d = 0
    if pd.isna(gld_50d): gld_50d = 0

    if gld_20d > 3 and gld_50d > 5:
        s = 1.0;  bull.append("Gold ETFs seeing strong institutional buying over 1-3 months — smart money is accumulating gold")
    elif gld_20d > 1.5:
        s = 0.6;  bull.append("Institutional gold ETF demand is rising — a positive sign for gold direction")
    elif gld_20d < -3 and gld_50d < -5:
        s = -1.0; bear.append("Institutional investors are selling gold ETFs heavily — smart money is reducing gold exposure")
    elif gld_20d < -1.5:
        s = -0.6; bear.append("Gold ETF demand is weakening — institutions are reducing positions")
    else:
        s = 0.0
    signals['GLD'] = s

    # Gold/Silver Ratio
    gs = row['GS_Ratio']
    if pd.isna(gs): gs = 80
    if gs > 90:
        s = -0.7; bear.append(f"Gold/Silver ratio is very high ({gs:.0f}) — gold is expensive relative to silver historically")
    elif gs > 80:
        s = -0.3; bear.append(f"Gold/Silver ratio is elevated ({gs:.0f}) — mild overvaluation signal")
    elif gs < 65:
        s = 0.7;  bull.append(f"Gold/Silver ratio is low ({gs:.0f}) — gold is attractively priced relative to silver")
    elif gs < 75:
        s = 0.3;  bull.append(f"Gold/Silver ratio is reasonable ({gs:.0f}) — no valuation concern")
    else:
        s = 0.0
    signals['GS_Ratio'] = s

    # Real Yield Proxy
    real_yield_proxy = tnx - max(0, gld_20d * 0.5)
    if real_yield_proxy < 1.5:
        s = 1.0;  bull.append("Real yields are very low — gold historically performs best in this environment")
    elif real_yield_proxy < 3.0:
        s = 0.4;  bull.append("Real yields are moderate — neutral to mildly supportive for gold")
    elif real_yield_proxy > 4.5:
        s = -1.0; bear.append("Real yields are high — cash and bonds offer better returns than gold right now")
    elif real_yield_proxy > 3.5:
        s = -0.4; bear.append("Real yields are above average — mild headwind for gold")
    else:
        s = 0.0
    signals['RealYield'] = s

    # DXY 30-Day Trend
    dxy_30d = row.get('DXY_30d_chg', 0)
    dxy     = row['DXY']
    if pd.isna(dxy_30d): dxy_30d = 0

    if dxy_30d < -2:
        s = 1.0;  bull.append(f"US Dollar has weakened significantly over the past month (DXY {dxy:.1f}) — a major tailwind for gold")
    elif dxy_30d < -0.5:
        s = 0.5;  bull.append(f"Dollar is on a modest downtrend (DXY {dxy:.1f}) — supportive for gold prices")
    elif dxy_30d > 2:
        s = -1.0; bear.append(f"US Dollar has strengthened significantly (DXY {dxy:.1f}) — a strong headwind for gold")
    elif dxy_30d > 0.5:
        s = -0.5; bear.append(f"Dollar has been ticking higher (DXY {dxy:.1f}) — mild pressure on gold")
    else:
        s = 0.0
    signals['DXY_Trend'] = s

    total_w   = sum(fund_weights.values())
    raw       = sum(signals.get(k, 0) * fund_weights[k] for k in fund_weights)
    score_100 = ((raw / total_w) + 1) / 2 * 100
    return round(score_100, 1), signals, bull, bear


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCORE & SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
def combined_score(tech_score, fund_score, tech_pct=60):
    fund_pct = 100 - tech_pct
    return round((tech_score * tech_pct + fund_score * fund_pct) / 100, 1)


def get_signal(score, buy_thresh, sell_thresh):
    if score >= buy_thresh:   return 'BUY'
    elif score <= sell_thresh: return 'SELL'
    return 'WAIT'


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(df, tech_weights, fund_weights, tech_pct, buy_thresh, sell_thresh):
    MIN_HOLD = 10
    CONFIRM  = 3

    records = []
    for i in range(len(df)):
        row      = df.iloc[i]
        prev_row = df.iloc[i-1] if i > 0 else None
        ts, _, _, _ = score_technical(row, prev_row, tech_weights)
        fs, _, _, _ = score_fundamental(row, prev_row, fund_weights)
        cs = combined_score(ts, fs, tech_pct)
        records.append({
            'Date':        df.index[i],
            'Close_INR':   row['Close_INR'],
            'Tech_Score':  ts,
            'Fund_Score':  fs,
            'Combined':    cs,
            'Above_SMA50': row['Close_INR'] > row['SMA50'],
        })

    res = pd.DataFrame(records).set_index('Date')

    confirmed = []
    for i in range(len(res)):
        if i < CONFIRM - 1:
            confirmed.append('WAIT'); continue
        w = res.iloc[i - CONFIRM + 1: i + 1]
        all_buy  = all(w['Combined'] >= buy_thresh) and all(w['Above_SMA50'])
        all_sell = all(w['Combined'] <= sell_thresh)
        confirmed.append('BUY' if all_buy else 'SELL' if all_sell else 'WAIT')

    res['Signal'] = confirmed

    trades = []
    in_trade = False
    entry_price = entry_date = 0
    entry_idx = 0

    for i in range(len(res)):
        sig   = res.iloc[i]['Signal']
        price = res.iloc[i]['Close_INR']
        date  = res.index[i]
        if not in_trade and sig == 'BUY':
            in_trade = True
            entry_price = price
            entry_date  = date
            entry_idx   = i
        elif in_trade and (i - entry_idx) >= MIN_HOLD and sig == 'SELL':
            pnl = price - entry_price
            trades.append({
                'Entry': entry_date, 'Exit': date,
                'Entry_Price': entry_price, 'Exit_Price': price,
                'Hold_Days': i - entry_idx,
                'PnL_INR': round(pnl, 2),
                'PnL_Pct': round(pnl / entry_price * 100, 2),
            })
            in_trade = False

    return res, pd.DataFrame(trades) if trades else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
TECH_W_DEF = {'RSI': 15, 'MACD': 15, 'MA': 15, 'ADX': 15,
              'BB': 10, 'Stoch': 10, 'ATR': 2}
# Note: MACD+RSI+MA+ADX+BB+Stoch+ATR = 82 → nudge BB/Stoch to reach 100
TECH_W_DEF = {'RSI': 15, 'MACD': 15, 'MA': 20, 'ADX': 15,
              'BB': 13, 'Stoch': 10, 'ATR': 2}   # sums to 90 — corrected below
TECH_W_DEF = {'RSI': 15, 'MACD': 15, 'MA': 20, 'ADX': 18,
              'BB': 12, 'Stoch': 10, 'ATR': 10}  # sums to 100

FUND_W_DEF = {'TNX': 35, 'GLD': 20, 'GS_Ratio': 20,
              'RealYield': 15, 'DXY_Trend': 10}   # sums to 100

for k, v in [('tech_w', TECH_W_DEF), ('fund_w', FUND_W_DEF),
             ('tech_pct', 60), ('buy_thresh', 62), ('sell_thresh', 38)]:
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, dict) else v


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🥇 MCX Gold Buying Advisor")
st.caption("Daily signals for physical gold timing | COMEX × USD/INR | 5-Year Backtest")

with st.spinner("Fetching 5 years of market data..."):
    df_raw = fetch_all_data(period="5y")

if df_raw is None or df_raw.empty:
    st.error("Could not fetch data. Check your internet connection.")
    st.stop()

df = compute_indicators(df_raw.copy())
df = add_rolling_features(df)

if df.empty:
    st.error("Not enough data after computing indicators.")
    st.stop()

latest = df.iloc[-1]
prev   = df.iloc[-2]

tech_score, tech_signals, tech_bull, tech_bear = score_technical(
    latest, prev, st.session_state.tech_w)
fund_score, fund_signals, fund_bull, fund_bear = score_fundamental(
    latest, prev, st.session_state.fund_w)

final_score = combined_score(tech_score, fund_score, st.session_state.tech_pct)
signal      = get_signal(final_score, st.session_state.buy_thresh, st.session_state.sell_thresh)

SIGNAL_MAP = {
    'BUY':  ('Good Time to Buy 🟢',          'signal-buy',  '#2ecc71'),
    'SELL': ('Consider Selling / Avoid 🔴',  'signal-sell', '#e74c3c'),
    'WAIT': ('Wait for Better Price 🟡',      'signal-wait', '#f39c12'),
}
sig_label, sig_class, sig_color = SIGNAL_MAP[signal]

all_bull = tech_bull + fund_bull
all_bear = tech_bear + fund_bear

price_chg     = latest['Close_INR'] - prev['Close_INR']
price_chg_pct = price_chg / prev['Close_INR'] * 100


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Today's Signal", "⚙️ Details & Weights", "🔁 Backtest (5 Years)"])


# ════════ TAB 1 ═══════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    arrow   = "▲" if price_chg >= 0 else "▼"
    chg_col = "#2ecc71" if price_chg >= 0 else "#e74c3c"

    with c1:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:12px;color:#aaa;'>MCX Gold (per 10g)</div>
            <div style='font-size:26px;font-weight:700;color:#f0d080;'>₹{latest['Close_INR']:,.0f}</div>
            <div style='font-size:13px;color:{chg_col};'>{arrow} ₹{abs(price_chg):,.0f} ({price_chg_pct:+.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:12px;color:#aaa;'>USD / INR</div>
            <div style='font-size:22px;font-weight:600;'>₹{latest['USDINR']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:12px;color:#aaa;'>US 10-Yr Yield</div>
            <div style='font-size:22px;font-weight:600;'>{latest['TNX']:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='price-box'>
            <div style='font-size:12px;color:#aaa;'>DXY (Dollar Index)</div>
            <div style='font-size:22px;font-weight:600;'>{latest['DXY']:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_sig, col_scores = st.columns([1, 1])

    with col_sig:
        st.markdown(f"""<div class='{sig_class}'>
            <div style='font-size:14px;color:#ccc;margin-bottom:8px;'>Overall Recommendation</div>
            <div style='font-size:34px;font-weight:900;color:{sig_color};'>{sig_label}</div>
            <div style='font-size:26px;font-weight:700;color:{sig_color};margin-top:8px;'>
                Score: {final_score}<span style='font-size:15px;color:#aaa;'>/100</span>
            </div>
            <div style='font-size:12px;color:#bbb;margin-top:6px;'>
                Buy ≥ {st.session_state.buy_thresh} | Sell ≤ {st.session_state.sell_thresh}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_scores:
        tc = "#2ecc71" if tech_score >= 55 else "#e74c3c" if tech_score <= 45 else "#f39c12"
        fc = "#2ecc71" if fund_score >= 55 else "#e74c3c" if fund_score <= 45 else "#f39c12"
        st.markdown(f"""<div class='score-block'>
            <div style='font-size:12px;color:#aaa;margin-bottom:2px;'>
                📈 Technical Score
                <span style='float:right;font-weight:700;color:{tc};'>{tech_score}/100</span>
            </div>
            <div class='bar-outer'><div style='width:{tech_score}%;background:{tc};height:100%;border-radius:10px;'></div></div>
            <div style='font-size:12px;color:#aaa;margin-bottom:2px;'>
                🌍 Fundamental Score
                <span style='float:right;font-weight:700;color:{fc};'>{fund_score}/100</span>
            </div>
            <div class='bar-outer'><div style='width:{fund_score}%;background:{fc};height:100%;border-radius:10px;'></div></div>
            <div style='font-size:12px;color:#aaa;margin-bottom:2px;'>
                ⚖️ Combined Score
                <span style='float:right;font-weight:700;color:{sig_color};'>{final_score}/100</span>
            </div>
            <div class='bar-outer'><div style='width:{final_score}%;background:{sig_color};height:100%;border-radius:10px;'></div></div>
            <div style='font-size:11px;color:#555;margin-top:2px;'>
                Technical {st.session_state.tech_pct}% · Fundamental {100 - st.session_state.tech_pct}%
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown("### ✅ Why this looks like a good time to buy")
        if all_bull:
            html = "".join(f"<p style='margin:6px 0;font-size:13px;'>✅ {b}</p>" for b in all_bull)
            st.markdown(f"<div class='pros-box'>{html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pros-box'><p style='color:#888;'>No strong bullish signals right now.</p></div>",
                        unsafe_allow_html=True)

    with col_c:
        st.markdown("### ⚠️ Reasons to wait or be cautious")
        if all_bear:
            html = "".join(f"<p style='margin:6px 0;font-size:13px;'>⚠️ {b}</p>" for b in all_bear)
            st.markdown(f"<div class='cons-box'>{html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='cons-box'><p style='color:#888;'>No significant bearish signals right now.</p></div>",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class='manual-box'>
        <div style='font-size:13px;color:#f0d080;font-weight:600;margin-bottom:8px;'>
            📋 Also consider before buying physical gold:
        </div>
        <div style='font-size:12px;color:#aaa;line-height:1.9;'>
        • <b>RBI / Monetary Policy</b> — Is RBI cutting or hiking rates?<br>
        • <b>India Import Duty</b> — Any recent changes to gold import tax or GST?<br>
        • <b>Seasonal Demand</b> — Wedding season (Oct–Dec, Apr–May) typically pushes prices higher<br>
        • <b>Geopolitical Events</b> — War, sanctions, or global uncertainty tends to spike gold<br>
        • <b>Central Bank Buying</b> — RBI or global CB accumulation is a long-term bullish signal
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Last updated: {df.index[-1].strftime('%d %b %Y')} | Refreshes every hour")
    if st.button("🔄 Refresh Data", key="refresh"):
        st.cache_data.clear()
        st.rerun()


# ════════ TAB 2 ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Technical Indicator Breakdown")
    tech_names = {
        'RSI': 'RSI (14)', 'MACD': 'MACD (12/26/9)', 'MA': 'Moving Averages',
        'ADX': 'ADX (14)', 'BB': 'Bollinger Bands', 'Stoch': 'Stochastic (14,3)', 'ATR': 'ATR'
    }
    tech_vals = {
        'RSI':   f"{latest['RSI']:.1f}",
        'MACD':  f"MACD {latest['MACD']:.0f} | Hist {latest['MACD_Hist']:.0f}",
        'MA':    f"SMA50 ₹{latest['SMA50']:,.0f} | SMA200 ₹{latest['SMA200']:,.0f}",
        'ADX':   f"{latest['ADX']:.1f} | +DI {latest['Plus_DI']:.1f} -DI {latest['Minus_DI']:.1f}",
        'BB':    f"BB% {latest['BB_Pct']*100:.1f}%",
        'Stoch': f"K {latest['Stoch_K']:.1f} | D {latest['Stoch_D']:.1f}",
        'ATR':   f"₹{latest['ATR']:,.0f} ({latest['ATR_Pct']:.2f}%)",
    }
    t_rows = []
    for k, name in tech_names.items():
        s = tech_signals.get(k, 0)
        sig_str = "Bullish ✅" if s > 0.3 else "Bearish 🔴" if s < -0.3 else "Neutral ➖"
        t_rows.append({'Indicator': name, 'Value': tech_vals[k], 'Signal': sig_str,
                       'Weight': f"{st.session_state.tech_w.get(k,0)}%"})
    st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

    st.markdown("### 🌍 Fundamental Factor Breakdown")
    fund_names = {
        'TNX': 'US 10-Yr Yield', 'GLD': 'Gold ETF Flow (GLD)',
        'GS_Ratio': 'Gold/Silver Ratio', 'RealYield': 'Real Yield Proxy',
        'DXY_Trend': 'DXY 30-Day Trend'
    }
    tnx_30d_val  = latest.get('TNX_30d_chg', 0)
    gld_20d_val  = latest.get('GLD_20d_chg', 0)
    dxy_30d_val  = latest.get('DXY_30d_chg', 0)
    tnx_30d_val  = 0 if pd.isna(tnx_30d_val)  else tnx_30d_val
    gld_20d_val  = 0 if pd.isna(gld_20d_val)  else gld_20d_val
    dxy_30d_val  = 0 if pd.isna(dxy_30d_val)  else dxy_30d_val
    real_yld_val = latest['TNX'] - max(0, gld_20d_val * 0.5)
    fund_vals = {
        'TNX':       f"{latest['TNX']:.2f}% (30d Δ: {tnx_30d_val:+.2f}%)",
        'GLD':       f"20d change: {gld_20d_val:+.1f}%",
        'GS_Ratio':  f"{latest['GS_Ratio']:.1f}x",
        'RealYield': f"Proxy: {real_yld_val:.2f}",
        'DXY_Trend': f"DXY {latest['DXY']:.1f} (30d Δ: {dxy_30d_val:+.2f})",
    }
    f_rows = []
    for k, name in fund_names.items():
        s = fund_signals.get(k, 0)
        sig_str = "Bullish ✅" if s > 0.3 else "Bearish 🔴" if s < -0.3 else "Neutral ➖"
        f_rows.append({'Factor': name, 'Value': fund_vals[k], 'Signal': sig_str,
                       'Weight': f"{st.session_state.fund_w.get(k,0)}%"})
    st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### ⚖️ Technical vs Fundamental Split")
    st.session_state.tech_pct = st.slider(
        "Technical weight % (remainder = Fundamental)",
        30, 80, st.session_state.tech_pct, step=5)
    st.caption(f"Technical: {st.session_state.tech_pct}% | Fundamental: {100 - st.session_state.tech_pct}%")

    st.markdown("---")
    st.markdown("### 🔧 Technical Weights (must sum to 100%)")
    tw = st.session_state.tech_w
    tc1, tc2 = st.columns(2)
    with tc1:
        tw['RSI']   = st.slider("RSI",             0, 30, tw['RSI'],   step=1, key='t_rsi')
        tw['MACD']  = st.slider("MACD",            0, 30, tw['MACD'],  step=1, key='t_macd')
        tw['MA']    = st.slider("Moving Averages", 0, 30, tw['MA'],    step=1, key='t_ma')
        tw['ADX']   = st.slider("ADX",             0, 30, tw['ADX'],   step=1, key='t_adx')
    with tc2:
        tw['BB']    = st.slider("Bollinger Bands", 0, 20, tw['BB'],    step=1, key='t_bb')
        tw['Stoch'] = st.slider("Stochastic",      0, 20, tw['Stoch'], step=1, key='t_stoch')
        tw['ATR']   = st.slider("ATR",             0, 15, tw['ATR'],   step=1, key='t_atr')
    tw_total = sum(tw.values())
    st.warning(f"⚠️ Total: {tw_total}% — adjust to 100%") if tw_total != 100 else st.success("✅ Technical weights = 100%")

    st.markdown("### 🌍 Fundamental Weights (must sum to 100%)")
    fw = st.session_state.fund_w
    fc1, fc2 = st.columns(2)
    with fc1:
        fw['TNX']      = st.slider("US 10-Yr Yield",    0, 50, fw['TNX'],      step=1, key='f_tnx')
        fw['GLD']      = st.slider("Gold ETF Flow",     0, 35, fw['GLD'],      step=1, key='f_gld')
        fw['GS_Ratio'] = st.slider("Gold/Silver Ratio", 0, 30, fw['GS_Ratio'], step=1, key='f_gs')
    with fc2:
        fw['RealYield'] = st.slider("Real Yield Proxy",  0, 30, fw['RealYield'], step=1, key='f_ry')
        fw['DXY_Trend'] = st.slider("DXY 30-Day Trend",  0, 20, fw['DXY_Trend'], step=1, key='f_dxy')
    fw_total = sum(fw.values())
    st.warning(f"⚠️ Total: {fw_total}% — adjust to 100%") if fw_total != 100 else st.success("✅ Fundamental weights = 100%")

    st.markdown("---")
    st.markdown("### 🎯 Signal Thresholds")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.session_state.buy_thresh  = st.slider("Buy threshold (score ≥)",  50, 85, st.session_state.buy_thresh,  step=1)
    with bc2:
        st.session_state.sell_thresh = st.slider("Sell threshold (score ≤)", 15, 50, st.session_state.sell_thresh, step=1)

    if st.button("✅ Apply & Recalculate", key="apply"):
        st.session_state.tech_w = tw
        st.session_state.fund_w = fw
        st.rerun()


# ════════ TAB 3 ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔁 Backtest — 5 Years of Daily Data")
    st.info("""**Active filters:**
✅ Trend filter — BUY only when price is above SMA50
✅ 3-day confirmation — score must stay above threshold for 3 consecutive days
✅ Minimum hold — 10 days before any exit is considered""")

    if st.button("▶️ Run Backtest", key="run_bt"):
        with st.spinner("Running 5-year backtest on daily candles..."):
            bt_res, trades_df = run_backtest(
                df,
                st.session_state.tech_w,
                st.session_state.fund_w,
                st.session_state.tech_pct,
                st.session_state.buy_thresh,
                st.session_state.sell_thresh
            )

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=("MCX Gold Price INR/10g",
                                            "Combined Score", "Technical vs Fundamental"))

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open_INR'], high=df['High_INR'],
            low=df['Low_INR'], close=df['Close_INR'], name="Gold",
            increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
            line=dict(color='#f0d080', width=1, dash='dot'), name='SMA50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'],
            line=dict(color='#9b59b6', width=1, dash='dot'), name='SMA200'), row=1, col=1)

        buy_idx   = bt_res[bt_res['Signal'] == 'BUY'].index
        sell_idx  = bt_res[bt_res['Signal'] == 'SELL'].index
        buy_valid  = buy_idx[buy_idx.isin(df.index)]
        sell_valid = sell_idx[sell_idx.isin(df.index)]

        fig.add_trace(go.Scatter(x=buy_valid, y=df.loc[buy_valid, 'Low_INR'] * 0.998,
            mode='markers', marker=dict(symbol='triangle-up', size=10, color='#2ecc71'),
            name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_valid, y=df.loc[sell_valid, 'High_INR'] * 1.002,
            mode='markers', marker=dict(symbol='triangle-down', size=10, color='#e74c3c'),
            name='Sell Signal'), row=1, col=1)

        fig.add_trace(go.Scatter(x=bt_res.index, y=bt_res['Combined'],
            line=dict(color='#f0d080', width=1.5), name='Combined'), row=2, col=1)
        fig.add_hline(y=st.session_state.buy_thresh,  line_dash="dot", line_color="#2ecc71", row=2, col=1)
        fig.add_hline(y=st.session_state.sell_thresh, line_dash="dot", line_color="#e74c3c", row=2, col=1)

        fig.add_trace(go.Scatter(x=bt_res.index, y=bt_res['Tech_Score'],
            line=dict(color='#3498db', width=1), name='Technical'), row=3, col=1)
        fig.add_trace(go.Scatter(x=bt_res.index, y=bt_res['Fund_Score'],
            line=dict(color='#e67e22', width=1), name='Fundamental'), row=3, col=1)

        fig.update_layout(
            template='plotly_dark', height=750,
            paper_bgcolor='#0f0f0f', plot_bgcolor='#0f0f0f',
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor='#1a1a2e', bordercolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Buy Signals",  len(buy_idx))
        m2.metric("Sell Signals", len(sell_idx))

        if not trades_df.empty:
            wins      = trades_df[trades_df['PnL_INR'] > 0]
            losses    = trades_df[trades_df['PnL_INR'] <= 0]
            win_rate  = len(wins) / len(trades_df) * 100
            total_pnl = trades_df['PnL_INR'].sum()

            m3.metric("Win Rate",  f"{win_rate:.1f}%")
            m4.metric("Total P&L", f"₹{total_pnl:,.0f}",
                      delta=f"{'▲' if total_pnl > 0 else '▼'} {abs(total_pnl):,.0f}")

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total Trades",  len(trades_df))
            e2.metric("Avg Hold",      f"{trades_df['Hold_Days'].mean():.0f} days")
            e3.metric("Avg Win",       f"{wins['PnL_Pct'].mean():.2f}%" if len(wins) > 0 else "—")
            e4.metric("Avg Loss",      f"{losses['PnL_Pct'].mean():.2f}%" if len(losses) > 0 else "—")

            st.markdown("### 🧾 Trade Log")
            td = trades_df.copy()
            td['Entry_Price'] = td['Entry_Price'].apply(lambda x: f"₹{x:,.0f}")
            td['Exit_Price']  = td['Exit_Price'].apply(lambda x: f"₹{x:,.0f}")
            td['PnL_INR']     = td['PnL_INR'].apply(lambda x: f"₹{x:,.0f}")
            td['PnL_Pct']     = td['PnL_Pct'].apply(lambda x: f"{x:+.2f}%")
            td['Hold_Days']   = td['Hold_Days'].apply(lambda x: f"{x}d")
            td.columns = ['Entry Date', 'Exit Date', 'Entry Price',
                          'Exit Price', 'Hold', 'P&L (₹)', 'P&L (%)']
            st.dataframe(td, use_container_width=True, hide_index=True)

            rr = abs(wins['PnL_Pct'].mean() / losses['PnL_Pct'].mean()) if len(losses) > 0 else 0
            st.markdown("### 💡 Insights")
            st.info(f"""
**Win Rate:** {win_rate:.1f}% | **Trades:** {len(trades_df)} | **Risk:Reward:** {rr:.1f}x | **Total P&L:** ₹{total_pnl:,.0f}
**Thresholds:** Buy ≥ {st.session_state.buy_thresh} | Sell ≤ {st.session_state.sell_thresh}
Adjust weights & thresholds in Tab 2, then re-run to optimise.""")
        else:
            m3.metric("Completed Trades", "0")
            m4.metric("Total P&L", "₹0")
            st.warning("No completed trades found. Try lowering the Buy threshold or raising the Sell threshold in Tab 2.")
