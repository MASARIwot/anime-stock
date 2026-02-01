"""Streamlit dashboard for Animetrics AI."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import requests

# Must be first Streamlit command
st.set_page_config(
    page_title="Animetrics AI",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import after st.set_page_config
from anime_stock.database.repositories import (
    TickerRepository,
    PriceRepository,
    SentimentRepository,
    PredictionRepository,
    ExchangeRateRepository,
    NewsRepository,
)


# --- STYLING (Light Theme) ---
THEME_CSS = """
<style>
    /* Light clean theme to match website */
    .stApp {
        background-color: #ffffff;
    }
    
    .main-header {
        color: #333333;
        border-bottom: 2px solid #e94560;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .index-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .index-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e94560;
    }
    
    .index-change-up {
        color: #28a745;
        font-size: 1.2rem;
    }
    
    .index-change-down {
        color: #dc3545;
        font-size: 1.2rem;
    }
    
    .news-item {
        padding: 10px 0;
        border-bottom: 1px solid #eee;
    }
    
    .news-source {
        color: #6c757d;
        font-size: 0.8rem;
    }
    
    .news-title {
        color: #333;
        font-weight: 500;
    }
    
    .news-title a {
        color: #e94560;
        text-decoration: none;
    }
    
    .news-title a:hover {
        text-decoration: underline;
    }
    
    h1, h2, h3 {
        color: #333333 !important;
    }
    
    .prediction-up {
        color: #28a745 !important;
        font-weight: bold;
    }
    
    .prediction-down {
        color: #dc3545 !important;
        font-weight: bold;
    }
</style>
"""


# --- DATA LOADING ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_tickers() -> list[dict]:
    """Load all active tickers."""
    tickers = TickerRepository.get_all_active()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "name": t.company_name,
            "currency": t.currency,
            "sector": t.sector,
        }
        for t in tickers
    ]


@st.cache_data(ttl=300)
def load_prices(ticker_id: int) -> pd.DataFrame:
    """Load price data for a ticker."""
    prices = PriceRepository.get_prices_for_ticker(ticker_id)
    if not prices:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            "date": p.date,
            "open": float(p.open) if p.open else None,
            "high": float(p.high) if p.high else None,
            "low": float(p.low) if p.low else None,
            "close": float(p.close),
            "volume": p.volume or 0,
        }
        for p in prices
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    # Add technical indicators
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    
    return df


@st.cache_data(ttl=300)
def load_all_prices() -> pd.DataFrame:
    """Load all prices for all tickers."""
    rows = PriceRepository.get_all_prices_with_ticker()
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    return df


@st.cache_data(ttl=300)
def load_sentiments() -> pd.DataFrame:
    """Load all sentiment scores."""
    scores = SentimentRepository.get_all_scores()
    if not scores:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {"date": s.date, "sentiment": float(s.score), "headlines": s.headlines_count}
        for s in scores
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=300)
def load_latest_news(limit: int = 10) -> list[dict]:
    """Load latest news articles."""
    articles = NewsRepository.get_latest_articles(limit)
    return [
        {
            "source": a.source,
            "title": a.title,
            "url": a.url,
            "published_at": a.published_at,
        }
        for a in articles
    ]


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_exchange_rate() -> float:
    """Get current USD/JPY exchange rate."""
    try:
        # Try database first
        cached = ExchangeRateRepository.get_latest_rate("USD", "JPY")
        if cached:
            return cached
        
        # Fetch from API
        response = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=JPY",
            timeout=5
        )
        if response.ok:
            rate = response.json()["rates"]["JPY"]
            ExchangeRateRepository.insert_rate("USD", "JPY", rate, date.today())
            return rate
    except Exception:
        pass
    
    return 150.0  # Fallback


def convert_price(price: float, from_currency: str, to_currency: str, rate: float) -> float:
    """Convert price between currencies."""
    if from_currency == to_currency:
        return price
    
    if from_currency == "JPY" and to_currency == "USD":
        return price / rate
    elif from_currency == "USD" and to_currency == "JPY":
        return price * rate
    
    return price


@st.cache_data(ttl=300)
def calculate_anime_index(display_currency: str, rate: float) -> pd.DataFrame:
    """
    Calculate a composite Anime Index across all tracked stocks.
    Normalizes each stock to 100 at the start date and averages them.
    """
    all_prices = load_all_prices()
    tickers = load_tickers()
    
    if all_prices.empty:
        return pd.DataFrame()
    
    # Create ticker lookup for currency
    ticker_currency = {t["symbol"]: t["currency"] for t in tickers}
    
    # Pivot to get each ticker as a column
    pivot = all_prices.pivot_table(
        index="date", 
        columns="symbol", 
        values="close", 
        aggfunc="first"
    )
    
    # Convert all to the display currency
    for col in pivot.columns:
        orig_currency = ticker_currency.get(col, "USD")
        if orig_currency != display_currency:
            pivot[col] = pivot[col].apply(
                lambda x: convert_price(x, orig_currency, display_currency, rate) if pd.notna(x) else x
            )
    
    # Normalize each column to 100 at the first non-null value
    normalized = pd.DataFrame(index=pivot.index)
    for col in pivot.columns:
        first_valid = pivot[col].first_valid_index()
        if first_valid is not None:
            base_value = pivot.loc[first_valid, col]
            if base_value > 0:
                normalized[col] = (pivot[col] / base_value) * 100
    
    # Calculate the average index
    normalized["Anime Index"] = normalized.mean(axis=1)
    
    # Forward fill any gaps
    normalized = normalized.ffill()
    
    return normalized


# --- CHARTS ---
def create_index_chart(index_df: pd.DataFrame, sentiments: pd.DataFrame) -> go.Figure:
    """Create the Anime Index chart with sentiment overlay."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
        subplot_titles=("Anime Industry Index (Normalized to 100)", "Market Sentiment"),
    )
    
    # Main index line
    fig.add_trace(
        go.Scatter(
            x=index_df.index,
            y=index_df["Anime Index"],
            name="Anime Index",
            line=dict(color="#e94560", width=3),
            fill="tozeroy",
            fillcolor="rgba(233, 69, 96, 0.1)",
        ),
        row=1, col=1,
    )
    
    # Add individual stock lines (faded)
    colors = ["#6c757d", "#adb5bd", "#dee2e6", "#868e96", "#495057"]
    for i, col in enumerate([c for c in index_df.columns if c != "Anime Index"][:5]):
        fig.add_trace(
            go.Scatter(
                x=index_df.index,
                y=index_df[col],
                name=col,
                line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
                opacity=0.5,
            ),
            row=1, col=1,
        )
    
    # Sentiment bars (if available)
    if not sentiments.empty:
        sent_filtered = sentiments[
            (sentiments.index >= index_df.index.min()) & 
            (sentiments.index <= index_df.index.max())
        ]
        
        if not sent_filtered.empty:
            colors = ["#28a745" if s >= 0 else "#dc3545" for s in sent_filtered["sentiment"]]
            fig.add_trace(
                go.Bar(
                    x=sent_filtered.index,
                    y=sent_filtered["sentiment"],
                    name="Sentiment",
                    marker_color=colors,
                ),
                row=2, col=1,
            )
    
    # Layout - Light theme
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(t=50, b=20, l=50, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(color="#333333"),
    )
    
    fig.update_yaxes(title_text="Index Value", row=1, col=1, gridcolor="#e9ecef")
    fig.update_yaxes(title_text="Score", row=2, col=1, range=[-1.1, 1.1], gridcolor="#e9ecef")
    fig.update_xaxes(gridcolor="#e9ecef")
    
    return fig


def create_price_chart(
    df: pd.DataFrame,
    sentiments: pd.DataFrame,
    ticker_name: str,
    display_currency: str,
    ticker_currency: str,
    rate: float,
) -> go.Figure:
    """Create the price chart for a single stock."""
    
    # Convert prices if needed
    df = df.copy()
    if ticker_currency != display_currency:
        for col in ["open", "high", "low", "close", "sma_20", "sma_50"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: convert_price(x, ticker_currency, display_currency, rate) if pd.notna(x) else x
                )
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker_name} Price", "Volume", "Sentiment"),
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            name="Price",
            line=dict(color="#e94560", width=2),
        ),
        row=1, col=1,
    )
    
    # SMA lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sma_20"],
            name="SMA 20",
            line=dict(color="#17a2b8", width=1, dash="dash"),
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sma_50"],
            name="SMA 50",
            line=dict(color="#ffc107", width=1, dash="dash"),
        ),
        row=1, col=1,
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color="rgba(233, 69, 96, 0.4)",
        ),
        row=2, col=1,
    )
    
    # Sentiment bars (if available)
    if not sentiments.empty:
        sent_filtered = sentiments[
            (sentiments.index >= df.index.min()) & 
            (sentiments.index <= df.index.max())
        ]
        
        if not sent_filtered.empty:
            colors = ["#28a745" if s >= 0 else "#dc3545" for s in sent_filtered["sentiment"]]
            fig.add_trace(
                go.Bar(
                    x=sent_filtered.index,
                    y=sent_filtered["sentiment"],
                    name="Sentiment",
                    marker_color=colors,
                ),
                row=3, col=1,
            )
    
    # Layout - Light theme
    currency_symbol = "¥" if display_currency == "JPY" else "$"
    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=50, b=20, l=50, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(color="#333333"),
    )
    
    fig.update_yaxes(title_text=f"Price ({currency_symbol})", row=1, col=1, gridcolor="#e9ecef")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#e9ecef")
    fig.update_yaxes(title_text="Score", row=3, col=1, range=[-1.1, 1.1], gridcolor="#e9ecef")
    fig.update_xaxes(gridcolor="#e9ecef")
    
    return fig


# --- MAIN APP ---
def main():
    """Main Streamlit application."""
    
    # Apply custom CSS
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏯 Animetrics AI</h1>', unsafe_allow_html=True)
    st.markdown("*Anime Industry Market Intelligence with AI-Powered Sentiment Analysis*")
    
    # Load data
    tickers = load_tickers()
    
    if not tickers:
        st.warning("No tickers found. Please run the database migration and add tickers.")
        st.code("mysql -u melvoice -p melvoice < scripts/init_db.sql")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Currency toggle
    rate = get_exchange_rate()
    display_currency = st.sidebar.radio(
        "Display Currency",
        options=["USD", "JPY"],
        index=0,
        help=f"Current rate: 1 USD = {rate:.2f} JPY",
    )
    
    # Page selection
    page = st.sidebar.radio(
        "View",
        options=["📊 Anime Index", "📈 Individual Stocks"],
        index=0,
    )
    
    # Date range
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Date Range")
    date_range = st.sidebar.selectbox(
        "Period",
        options=["1M", "3M", "6M", "1Y", "2Y", "All"],
        index=3,
    )
    
    sentiments_df = load_sentiments()
    
    if page == "📊 Anime Index":
        # --- ANIME INDEX PAGE ---
        
        # Calculate index
        index_df = calculate_anime_index(display_currency, rate)
        
        if index_df.empty:
            st.warning("No price data available. Run the collector first.")
            return
        
        # Filter by date range
        if date_range != "All":
            days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_map[date_range])
            index_df = index_df[index_df.index >= cutoff]
        
        # --- TOP METRICS ROW ---
        col1, col2, col3, col4 = st.columns(4)
        
        # Current index value
        current_index = index_df["Anime Index"].iloc[-1]
        prev_index = index_df["Anime Index"].iloc[-2] if len(index_df) > 1 else current_index
        index_change = current_index - prev_index
        index_change_pct = (index_change / prev_index) * 100 if prev_index != 0 else 0
        
        with col1:
            st.metric(
                "Anime Index",
                f"{current_index:.1f}",
                f"{index_change_pct:+.2f}%",
            )
        
        # Latest sentiment
        if not sentiments_df.empty:
            latest_sentiment = sentiments_df["sentiment"].iloc[-1]
            sentiment_label = (
                "Bullish 🐂" if latest_sentiment > 0.3
                else "Bearish 🐻" if latest_sentiment < -0.3
                else "Neutral 😐"
            )
        else:
            latest_sentiment = 0.0
            sentiment_label = "No data"
        
        with col2:
            st.metric(
                "Market Sentiment",
                f"{latest_sentiment:.2f}",
                sentiment_label,
            )
        
        # Exchange rate
        with col3:
            st.metric(
                "USD/JPY",
                f"¥{rate:.2f}",
                "Live",
            )
        
        # Tracked stocks count
        with col4:
            st.metric(
                "Tracked Stocks",
                f"{len(tickers)}",
                "Active",
            )
        
        st.markdown("---")
        
        # --- MAIN CHART AND NEWS SIDE BY SIDE ---
        chart_col, news_col = st.columns([2, 1])
        
        with chart_col:
            st.subheader("📊 Anime Industry Composite Index")
            chart = create_index_chart(index_df, sentiments_df)
            st.plotly_chart(chart, use_container_width=True)
        
        with news_col:
            st.subheader("📰 Latest News")
            news_items = load_latest_news(10)
            
            if news_items:
                for item in news_items:
                    source_emoji = "🔵" if item["source"] == "ANN" else "🟢"
                    pub_date = item["published_at"].strftime("%b %d") if item["published_at"] else ""
                    
                    st.markdown(
                        f"""
                        <div class="news-item">
                            <span class="news-source">{source_emoji} {item['source']} • {pub_date}</span><br>
                            <span class="news-title"><a href="{item['url']}" target="_blank">{item['title'][:80]}{'...' if len(item['title']) > 80 else ''}</a></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No news articles yet. Run the news scraper.")
        
        st.markdown("---")
        
        # --- PREDICTIONS TABLE ---
        st.subheader("🤖 AI Predictions for Tomorrow")
        
        pred_data = []
        for ticker in tickers:
            prediction = PredictionRepository.get_latest_prediction(ticker["id"])
            if prediction:
                pred_data.append({
                    "Symbol": ticker["symbol"],
                    "Company": ticker["name"],
                    "Sector": ticker["sector"].capitalize(),
                    "Direction": f"{'📈' if prediction.direction == 'UP' else '📉'} {prediction.direction}",
                    "Confidence": f"{float(prediction.confidence):.0%}",
                })
        
        if pred_data:
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions available. Run the predictor first.")
    
    else:
        # --- INDIVIDUAL STOCKS PAGE ---
        
        # Ticker selector
        ticker_options = {f"{t['symbol']} - {t['name']}": t for t in tickers}
        selected = st.selectbox(
            "Select Stock",
            options=list(ticker_options.keys()),
        )
        selected_ticker = ticker_options[selected]
        
        # Load price data
        prices_df = load_prices(selected_ticker["id"])
        
        if prices_df.empty:
            st.warning(f"No price data for {selected_ticker['symbol']}. Run the collector first.")
            return
        
        # Filter by date range
        if date_range != "All":
            days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_map[date_range])
            prices_df = prices_df[prices_df.index >= cutoff]
        
        # --- METRICS ROW ---
        col1, col2, col3, col4 = st.columns(4)
        
        # Current price
        current_price = prices_df["close"].iloc[-1]
        prev_price = prices_df["close"].iloc[-2] if len(prices_df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        display_price = convert_price(
            current_price, 
            selected_ticker["currency"], 
            display_currency, 
            rate
        )
        currency_symbol = "¥" if display_currency == "JPY" else "$"
        
        with col1:
            st.metric(
                "Current Price",
                f"{currency_symbol}{display_price:,.2f}",
                f"{price_change:+.2f}%",
            )
        
        # Latest sentiment
        if not sentiments_df.empty:
            latest_sentiment = sentiments_df["sentiment"].iloc[-1]
        else:
            latest_sentiment = 0.0
        
        with col2:
            st.metric(
                "Market Sentiment",
                f"{latest_sentiment:.2f}",
            )
        
        # Prediction
        prediction = PredictionRepository.get_latest_prediction(selected_ticker["id"])
        if prediction:
            pred_direction = prediction.direction
            pred_confidence = float(prediction.confidence)
            pred_emoji = "📈" if pred_direction == "UP" else "📉"
        else:
            pred_direction = "N/A"
            pred_confidence = 0.0
            pred_emoji = "❓"
        
        with col3:
            st.metric(
                "AI Forecast",
                f"{pred_emoji} {pred_direction}",
                f"{pred_confidence:.0%} confidence" if prediction else "No prediction",
            )
        
        # 52-week range
        year_prices = prices_df["close"].tail(252)
        with col4:
            high_price = convert_price(year_prices.max(), selected_ticker["currency"], display_currency, rate)
            low_price = convert_price(year_prices.min(), selected_ticker["currency"], display_currency, rate)
            st.metric(
                "52W Range",
                f"{currency_symbol}{low_price:,.0f} - {currency_symbol}{high_price:,.0f}",
            )
        
        st.markdown("---")
        
        # --- MAIN CHART ---
        st.subheader(f"📊 {selected_ticker['name']} ({selected_ticker['symbol']})")
        
        chart = create_price_chart(
            prices_df,
            sentiments_df,
            selected_ticker["name"],
            display_currency,
            selected_ticker["currency"],
            rate,
        )
        st.plotly_chart(chart, use_container_width=True)
    
    # --- SIDEBAR INFO ---
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "Powered by Animetrics AI 🏯"
    )


def run():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    main()
