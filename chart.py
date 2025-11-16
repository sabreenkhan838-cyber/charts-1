import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
from io import BytesIO

st.set_page_config(layout="wide", page_title="Sensex Sector Analyzer")

SECTORS = {
    "Bank": "^BSEBANK",
    "Auto": "^BSEAUTO",
    "FMCG": "^BSEFMCG",
    "IT": "^BSEIT",
    "Healthcare": "^BSEHEALTH",
    "Metal": "^BSEMETAL",
}

@st.cache_data(ttl=300)
def fetch_df(symbol, period_days=180):
    return yf.download(symbol, period=f"{period_days}d", progress=False)

def plot_scatter_with_regression(close_series, title):
    current = close_series[:-1]
    future = close_series[1:]

    x = np.array(current).reshape(-1, 1)
    y = np.array(future)

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y, s=15)
    ax.plot(x, y_pred, lw=2)
    ax.set_title(title)
    ax.set_xlabel("Current Price")
    ax.set_ylabel("Next Day Price")
    ax.grid(True)
    return fig

def plot_candlestick(df):
    df2 = df.copy()
    df2.index.name = "Date"
    df2["MA50"] = df["Close"].rolling(50).mean()
    df2["MA100"] = df["Close"].rolling(100).mean()

    buf = BytesIO()
    mpf.plot(
        df2,
        type="candle",
        style="yahoo",
        mav=(50,100),
        volume=True,
        savefig=buf
    )
    buf.seek(0)
    return buf

def predict_next_day(series):
    df = pd.DataFrame({"Close": series})
    df["lag_1"] = df["Close"].shift(1)
    df["lag_2"] = df["Close"].shift(2)
    df.dropna(inplace=True)

    X = df[["lag_1", "lag_2"]].values
    y = df["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    last = df.tail(1)
    last_input = np.array([[last["lag_1"].values[0], last["lag_2"].values[0]]])
    return model.predict(last_input)[0]

# -------------------- UI --------------------

st.title("📊 Sensex Sector Dashboard")

sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))
days = st.sidebar.slider("Days of Data", 90, 720, 300)

df = fetch_df(SECTORS[sector], days)

st.subheader(f"{sector} Sector Data")
st.write(df.tail())

if df.empty:
    st.error("⚠ No data available for this sector.")
    st.stop()

close_series = df["Close"].dropna()

st.subheader("Scatter Plot with Regression")
fig = plot_scatter_with_regression(close_series, f"{sector} – Current vs Future Price")
st.pyplot(fig)

st.subheader("Candlestick Chart")
candle_img = plot_candlestick(df.tail(150))
st.image(candle_img)

st.subheader("Next-Day Price Prediction")
prediction = predict_next_day(close_series)
st.success(f"📈 Predicted Next-Day Close Price: **{prediction:.2f}**")

# Sensex Sector Dashboard

A Streamlit dashboard for analyzing Sensex sector indices with:

- Scatter plot (current vs next-day)
- Regression line
- Candlestick charts
- Moving averages (MA50, MA100)
- Next-day price prediction (Linear Regression)
- Sector selection
- Auto data fetch via yfinance

## How to Run (Locally)

