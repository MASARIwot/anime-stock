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
    initial_sidebar_state="collapsed",
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
from anime_stock.config import config
import mysql.connector
from anime_stock.dashboard.translations import get_text, format_date


# --- STYLING (Light Theme) ---
THEME_CSS = """
<style>
    /* Hide Streamlit elements for clean embed */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Remove top padding for iframe embed */
    .stApp > header {display: none;}
    .block-container {padding-top: 1rem !important;}
    [data-testid="stAppViewContainer"] {padding-top: 0 !important;}
    
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
def load_latest_news(limit: int = 50) -> list[dict]:
    """Load latest news articles for tracked stocks only."""
    # Request more articles to ensure we get enough with tickers
    articles = NewsRepository.get_latest_articles(limit)
    
    # Get tracked ticker symbols
    tickers = TickerRepository.get_all_active()
    tracked_symbols = {t.symbol for t in tickers}
    
    # Filter and return only news for tracked stocks (with ticker field populated)
    filtered = [
        {
            "source": a.source,
            "title": a.title,
            "url": a.url,
            "published_at": a.published_at,
            "ticker": a.ticker,
        }
        for a in articles
        if a.ticker and a.ticker in tracked_symbols
    ]
    
    # Return first 10 items
    return filtered[:10]


@st.cache_data(ttl=300)
def load_predictions_for_ticker(ticker_id: int) -> pd.DataFrame:
    """Load all predictions for a specific ticker."""
    conn = mysql.connector.connect(
        host=config.database.host,
        port=config.database.port,
        user=config.database.username,
        password=config.database.password,
        database=config.database.database,
    )
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT date, direction, actual_direction, confidence
            FROM predictions
            WHERE ticker_id = %s AND actual_direction IS NOT NULL
            ORDER BY date
        """, (ticker_id,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["confidence"] = df["confidence"].astype(float)
        
        return df
    finally:
        conn.close()


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
def create_index_chart(index_df: pd.DataFrame, sentiments: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Create the Anime Index chart with sentiment overlay."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(get_text("index_normalized", lang), get_text("chart_sentiment", lang)),
    )
    
    # Main index line
    fig.add_trace(
        go.Scatter(
            x=index_df.index,
            y=index_df["Anime Index"],
            name=get_text("anime_index", lang),
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
                    name=get_text("sentiment", lang),
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
        hovermode="x unified",
        dragmode=False,
    )
    
    fig.update_yaxes(title_text=get_text("index_value", lang), row=1, col=1, gridcolor="#e9ecef")
    fig.update_yaxes(title_text=get_text("score", lang), row=2, col=1, range=[-1.1, 1.1], gridcolor="#e9ecef")
    fig.update_xaxes(gridcolor="#e9ecef")
    
    return fig


def create_price_chart(
    df: pd.DataFrame,
    sentiments: pd.DataFrame,
    predictions: pd.DataFrame,
    ticker_name: str,
    display_currency: str,
    ticker_currency: str,
    rate: float,
    lang: str = "en",
) -> go.Figure:
    """Create the price chart for a single stock with AI predictions."""
    
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
        subplot_titles=(f"{ticker_name} {get_text('price', lang)}", get_text("volume", lang), get_text("sentiment", lang)),
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            name=get_text("price", lang),
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
    
    # Add AI Predictions as markers
    if not predictions.empty:
        pred_filtered = predictions[
            (predictions.index >= df.index.min()) & 
            (predictions.index <= df.index.max())
        ]
        
        if not pred_filtered.empty:
            # Get actual prices for prediction dates
            pred_with_price = pred_filtered.join(df[["close"]], how="inner")
            
            # Separate correct and incorrect predictions
            up_correct = pred_with_price[(pred_with_price["direction"] == "UP") & (pred_with_price["actual_direction"] == "UP")]
            up_wrong = pred_with_price[(pred_with_price["direction"] == "UP") & (pred_with_price["actual_direction"] == "DOWN")]
            down_correct = pred_with_price[(pred_with_price["direction"] == "DOWN") & (pred_with_price["actual_direction"] == "DOWN")]
            down_wrong = pred_with_price[(pred_with_price["direction"] == "DOWN") & (pred_with_price["actual_direction"] == "UP")]
            
            # UP predictions - correct (green triangle up)
            if not up_correct.empty:
                fig.add_trace(
                    go.Scatter(
                        x=up_correct.index,
                        y=up_correct["close"],
                        mode="markers",
                        name="✅ AI: UP (Correct)",
                        marker=dict(symbol="triangle-up", size=12, color="#28a745", line=dict(width=2, color="white")),
                        hovertemplate="<b>AI Predicted: UP</b><br>Result: UP ✅<br>Confidence: %{customdata:.0%}<extra></extra>",
                        customdata=up_correct["confidence"],
                    ),
                    row=1, col=1,
                )
            
            # UP predictions - wrong (gray triangle up)
            if not up_wrong.empty:
                fig.add_trace(
                    go.Scatter(
                        x=up_wrong.index,
                        y=up_wrong["close"],
                        mode="markers",
                        name="❌ AI: UP (Wrong)",
                        marker=dict(symbol="triangle-up", size=12, color="#6c757d", line=dict(width=2, color="white")),
                        hovertemplate="<b>AI Predicted: UP</b><br>Result: DOWN ❌<br>Confidence: %{customdata:.0%}<extra></extra>",
                        customdata=up_wrong["confidence"],
                    ),
                    row=1, col=1,
        hovermode="x unified",
        dragmode=False,
                )
            
            # DOWN predictions - correct (red triangle down)
            if not down_correct.empty:
                fig.add_trace(
                    go.Scatter(
                        x=down_correct.index,
                        y=down_correct["close"],
                        mode="markers",
                        name="✅ AI: DOWN (Correct)",
                        marker=dict(symbol="triangle-down", size=12, color="#dc3545", line=dict(width=2, color="white")),
                        hovertemplate="<b>AI Predicted: DOWN</b><br>Result: DOWN ✅<br>Confidence: %{customdata:.0%}<extra></extra>",
                        customdata=down_correct["confidence"],
                    ),
                    row=1, col=1,
                )
            
            # DOWN predictions - wrong (gray triangle down)
            if not down_wrong.empty:
                fig.add_trace(
                    go.Scatter(
                        x=down_wrong.index,
                        y=down_wrong["close"],
                        mode="markers",
                        name="❌ AI: DOWN (Wrong)",
                        marker=dict(symbol="triangle-down", size=12, color="#6c757d", line=dict(width=2, color="white")),
                        hovertemplate="<b>AI Predicted: DOWN</b><br>Result: UP ❌<br>Confidence: %{customdata:.0%}<extra></extra>",
                        customdata=down_wrong["confidence"],
                    ),
                    row=1, col=1,
                )
    
    # 
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name=get_text("volume", lang),
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
                    name=get_text("sentiment", lang),
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
    
    fig.update_yaxes(title_text=f"{get_text('price', lang)} ({currency_symbol})", row=1, col=1, gridcolor="#e9ecef")
    fig.update_yaxes(title_text=get_text("volume", lang), row=2, col=1, gridcolor="#e9ecef")
    fig.update_yaxes(title_text=get_text("score", lang), row=3, col=1, range=[-1.1, 1.1], gridcolor="#e9ecef")
    fig.update_xaxes(gridcolor="#e9ecef")
    
    return fig


# --- MAIN APP ---
def main():
    """Main Streamlit application."""
    
    # Apply custom CSS
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    # Initialize language in session state
    if "lang" not in st.session_state:
        st.session_state.lang = "uk"
    
    # Load data
    tickers = load_tickers()
    
    if not tickers:
        st.warning(get_text("no_tickers", st.session_state.lang))
        return
    
    # Compact control bar - single row with dropdowns
    rate = get_exchange_rate()
    sentiments_df = load_sentiments()
    
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 3, 1])
    with c1:
        display_currency = st.selectbox(get_text("currency", st.session_state.lang), ["USD", "JPY"], label_visibility="collapsed")
    with c2:
        page = st.selectbox(
            get_text("view", st.session_state.lang),
            [get_text("view_index", st.session_state.lang), get_text("view_stocks", st.session_state.lang)],
            label_visibility="collapsed"
        )
    with c3:
        date_range = st.selectbox(get_text("period", st.session_state.lang), ["1M", "3M", "6M", "1Y", "2Y"], index=3, label_visibility="collapsed")
    with c4:
        st.markdown(f"💱 {display_currency} | 📅 {date_range} | ¥{rate:.0f} | {len(tickers)} {get_text('stocks', st.session_state.lang)}")
    with c5:
        lang_option = st.selectbox(
            "Language",
            options=["🇺🇦 UA", "🇺🇸 EN"],
            index=0 if st.session_state.lang == "uk" else 1,
            label_visibility="collapsed"
        )
        st.session_state.lang = "uk" if "UA" in lang_option else "en"
    
    # Normalize page selection to English for comparison
    page_normalized = "Index" if page == get_text("view_index", st.session_state.lang) else "Stocks"
    
    if page_normalized == "Index":
        # --- ANIME INDEX PAGE ---
        
        # Calculate index
        index_df = calculate_anime_index(display_currency, rate)
        
        if index_df.empty:
            st.warning(get_text("no_price_data", st.session_state.lang))
            return
        
        # Filter by date range
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
                get_text("anime_index", st.session_state.lang),
                f"{current_index:.1f}",
                f"{index_change_pct:+.2f}%",
            )
        
        # Latest sentiment
        if not sentiments_df.empty:
            latest_sentiment = sentiments_df["sentiment"].iloc[-1]
            sentiment_label = (
                get_text("sentiment_bullish", st.session_state.lang) if latest_sentiment > 0.3
                else get_text("sentiment_bearish", st.session_state.lang) if latest_sentiment < -0.3
                else get_text("sentiment_neutral", st.session_state.lang)
            )
        else:
            latest_sentiment = 0.0
            sentiment_label = get_text("sentiment_no_data", st.session_state.lang)
        
        with col2:
            st.metric(
                get_text("market_sentiment", st.session_state.lang),
                f"{latest_sentiment:.2f}",
                sentiment_label,
            )
        
        # Exchange rate
        with col3:
            st.metric(
                get_text("usd_jpy", st.session_state.lang),
                f"¥{rate:.2f}",
                get_text("live", st.session_state.lang),
            )
        
        # Tracked stocks count
        with col4:
            st.metric(
                get_text("tracked_stocks", st.session_state.lang),
                f"{len(tickers)}",
                get_text("active", st.session_state.lang),
            )
        
        st.markdown("---")
        
        # --- MAIN CHART AND NEWS SIDE BY SIDE ---
        chart_col, news_col = st.columns([2, 1])
        
        with chart_col:
            st.subheader(get_text("anime_industry_index", st.session_state.lang))
            chart = create_index_chart(index_df, sentiments_df, st.session_state.lang)
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
        
        with news_col:
            st.subheader(get_text("latest_news", st.session_state.lang))
            news_items = load_latest_news(10)
            
            if news_items:
                for item in news_items:
                    # Display ticker symbol prominently
                    ticker_tag = f"<b>[{item.get('ticker', '???')}]</b> " if item.get('ticker') else ""
                    pub_date = format_date(item["published_at"], st.session_state.lang, "short") if item["published_at"] else ""
                    
                    st.markdown(
                        f"""
                        <div class="news-item">
                            <span class="news-source">📈 {ticker_tag}{item['source']} • {pub_date}</span><br>
                            <span class="news-title"><a href="{item['url']}" target="_blank">{item['title'][:80]}{'...' if len(item['title']) > 80 else ''}</a></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(get_text("no_news", st.session_state.lang))
        
        st.markdown("---")
        
        # --- PREDICTIONS TABLE ---
        st.subheader(get_text("ai_predictions", st.session_state.lang))
        
        pred_data = []
        for ticker in tickers:
            prediction = PredictionRepository.get_latest_prediction(ticker["id"])
            if prediction:
                pred_data.append({
                    get_text("symbol", st.session_state.lang): ticker["symbol"],
                    get_text("company", st.session_state.lang): ticker["name"],
                    get_text("sector", st.session_state.lang): ticker["sector"].capitalize(),
                    get_text("direction", st.session_state.lang): f"{'📈' if prediction.direction == 'UP' else '📉'} {prediction.direction}",
                    get_text("table_confidence", st.session_state.lang): f"{float(prediction.confidence):.0%}",
                })
        
        if pred_data:
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.info(get_text("no_predictions", st.session_state.lang))
    
    else:
        # --- INDIVIDUAL STOCKS PAGE ---
        
        # Ticker selector
        ticker_options = {f"{t['symbol']} - {t['name']}": t for t in tickers}
        selected = st.selectbox(
            get_text("select_stock", st.session_state.lang),
            options=list(ticker_options.keys()),
        )
        selected_ticker = ticker_options[selected]
        
        # Load price data
        prices_df = load_prices(selected_ticker["id"])
        
        if prices_df.empty:
            st.warning(get_text("no_price_data_ticker", st.session_state.lang).format(symbol=selected_ticker['symbol']))
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
                get_text("current_price", st.session_state.lang),
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
                get_text("market_sentiment", st.session_state.lang),
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
                get_text("ai_forecast", st.session_state.lang),
                f"{pred_emoji} {pred_direction}",
                f"{pred_confidence:.0%} {get_text('confidence', st.session_state.lang)}" if prediction else get_text("no_prediction", st.session_state.lang),
            )
        
        # 52-week range
        year_prices = prices_df["close"].tail(252)
        with col4:
            high_price = convert_price(year_prices.max(), selected_ticker["currency"], display_currency, rate)
            low_price = convert_price(year_prices.min(), selected_ticker["currency"], display_currency, rate)
            st.metric(
                get_text("week_range", st.session_state.lang),
                f"{currency_symbol}{low_price:,.0f} - {currency_symbol}{high_price:,.0f}",
            )
        
        st.markdown("---")
        
        # --- MAIN CHART ---
        st.subheader(f"📊 {selected_ticker['name']} ({selected_ticker['symbol']})")
        
        # Load predictions for this ticker
        predictions_df = load_predictions_for_ticker(selected_ticker["id"])
        
        chart = create_price_chart(
            prices_df,
            sentiments_df,
            predictions_df,
            selected_ticker["name"],
            display_currency,
            selected_ticker["currency"],
            rate,
            st.session_state.lang,
        )
        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
    
    # --- FOOTER INFO ---
    st.markdown("---")
    timestamp = format_date(pd.Timestamp.now(), st.session_state.lang, "full")
    st.caption(
        f"{get_text('last_updated', st.session_state.lang)}: {timestamp} | "
        f"{get_text('powered_by', st.session_state.lang)}"
    )


def run():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    main()
