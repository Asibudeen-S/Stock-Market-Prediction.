# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import numpy as np
import requests
import plotly.graph_objects as go

# Try importing snscrape safely
try:
    import snscrape.modules.twitter as sntwitter
    SN_AVAILABLE = True
except ImportError:
    SN_AVAILABLE = False

# ---------------- PAGE SETTINGS ---------------- #
st.set_page_config(page_title="Stock Insights", layout="wide")
st.title("📈 Stock Insights & Prediction")

# ---------------- USER INPUTS ---------------- #
ticker = st.text_input("Enter Stock Symbol", value="AAPL")
start = st.date_input("Start Date", date(2015, 1, 1))
end = st.date_input("End Date", date.today())

# ---------------- FUNCTION: CLEAN COLUMNS ---------------- #
def clean_columns(df):
    """Flatten or clean column names if yfinance returns MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            df.columns = [
                "_".join([str(c) for c in col if c]).strip() for col in df.columns.values
            ]
    return df

# ---------------- CREATE TABS ---------------- #
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price Prediction", "📰 News", "🐦 Tweets", "📈 Technical Updates"]
)

# ---------------- TAB 1: PRICE PREDICTION ---------------- #
with tab1:
    if st.button("Predict"):
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error("No stock data found. Please check the symbol or date range.")
        else:
            data = clean_columns(data)
            if "Close" not in data.columns:
                st.error("Data does not have 'Close' column. Try another stock.")
            else:
                data["Prediction"] = data["Close"].shift(-1)  # Next day's close
                st.line_chart(data["Close"])

                X = data[["Close"]][:-1]
                y = data["Prediction"][:-1]

                if len(X) > 1:
                    model = LinearRegression().fit(X, y)
                    last_close = data["Close"].iloc[-1]
                    last_close_reshaped = np.array(last_close).reshape(1, -1)
                    next_price = model.predict(last_close_reshaped)[0]

                    st.subheader(f"Predicted Next Close: {next_price:.2f}")
                    st.write(data.tail())
                else:
                    st.warning("Not enough data to make a prediction.")

# ---------------- TAB 2: NEWS ---------------- #
# TAB 2: NEWS using Stock News API
with tab2:
    st.subheader(f"Latest Stock News for {ticker}")
    API_KEY = "468771d066e447f0990f748172e495b2"  # Replace with your real key

    if API_KEY == "STOCK_NEWS_API_KEY":
        st.info("⚠️ Please set your Stock News API key.")
    else:
        try:
            url = (
                f"https://stocknewsapi.com/api/v1?tickers={ticker}"
                f"&items=5&token={API_KEY}"
            )
            resp = requests.get(url).json()
            if resp.get("data"):
                for article in resp["data"]:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.caption(article.get("snip") or "")
                    sentiment = article.get("sentiment")
                    if sentiment:
                        st.write(f"*Sentiment:* {sentiment.capitalize()}")
                    st.write("---")
            else:
                st.warning("No news found.")
        except Exception as e:
            st.error(f"Error fetching news: {e}")


# ---------------- TAB 3: TWEETS ---------------- #
with tab3:
    st.subheader(f"Recent Tweets about {ticker}")

    if not SN_AVAILABLE:
        st.error("snscrape is not installed. Run: pip install snscrape")
    else:
        try:
            tweets_list = []
            since_date = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            query = f"{ticker} lang:en since:{since_date}"

            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= 10:
                    break
                tweets_list.append(
                    {
                        "Date": tweet.date.strftime("%Y-%m-%d %H:%M"),
                        "User": tweet.user.username,
                        "Tweet": tweet.content,
                    }
                )

            if tweets_list:
                tweets_df = pd.DataFrame(tweets_list)
                for _, row in tweets_df.iterrows():
                    st.markdown(f"**@{row['User']}** — {row['Date']}")
                    st.write(row["Tweet"])
                    st.write("---")
            else:
                st.warning("No recent tweets found.")
        except Exception as e:
            st.error(f"Error fetching tweets: {e}")

# ---------------- TAB 4: TECHNICAL UPDATES ---------------- #
with tab4:
    st.subheader("Technical Indicators")

    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        st.error("No stock data found for technical indicators.")
    else:
        data = clean_columns(data)
        if "Close" not in data.columns:
            st.error("Data does not have 'Close' column.")
        else:
            # Calculate SMAs
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["SMA_100"] = data["Close"].rolling(window=100).mean()
            data["SMA_150"] = data["Close"].rolling(window=150).mean()
            data.dropna(inplace=True)

            # Plot with Plotly
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="white", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["SMA_50"],
                    mode="lines",
                    name="SMA 50",
                    line=dict(color="red", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["SMA_100"],
                    mode="lines",
                    name="SMA 100",
                    line=dict(color="cyan", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["SMA_150"],
                    mode="lines",
                    name="SMA 150",
                    line=dict(color="magenta", width=2),
                )
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                legend_title="Legend",
                plot_bgcolor="black",
                paper_bgcolor="black",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Market Signal
            sma50 = data["SMA_50"].iloc[-1]
            sma100 = data["SMA_100"].iloc[-1]
            sma150 = data["SMA_150"].iloc[-1]

            st.write("### Market Signals")
            if sma50 > sma100 > sma150:
                st.success("📈 Strong bullish trend — Short-term > Medium-term > Long-term")
            elif sma50 < sma100 < sma150:
                st.error("📉 Strong bearish trend — Short-term < Medium-term < Long-term")
            else:
                st.warning("⚠️ Mixed or sideways trend — No clear signal")
