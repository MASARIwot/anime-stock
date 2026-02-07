"""Streamlit dashboard for Animetrics AI."""

import logging
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

# Initialize logger
logger = logging.getLogger(__name__)

# Calculate seconds until end of day for caching
def get_seconds_until_end_of_day():
    """Calculate seconds until end of current day for cache TTL."""
    from datetime import datetime, time
    now = datetime.now()
    end_of_day = datetime.combine(now.date(), time.max)
    return int((end_of_day - now).total_seconds())


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
    
    /* Clean light theme matching melvoice.com */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main .block-container {
        background-color: #ffffff !important;
        max-width: 1200px;
    }
    
    /* Force white background on all containers */
    .main, .block-container, [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
        background-color: #ffffff !important;
    }
    
    /* Clean text styling */
    h1, h2, h3 {
        color: #333 !important;
        font-weight: 500 !important;
    }
    
    /* Default text color (allow inline styles to override) */
    p, label {
        color: #333;
    }
    
    /* Subtle borders and spacing */
    .element-container {
        margin-bottom: 0.5rem;
        background-color: #ffffff !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-size: 13px;
    }
    
    /* Remove excessive spacing for iframe */
    .main {
        padding: 0 !important;
    }
    
    .news-item {
        padding: 10px 0;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .news-source {
        color: #666;
        font-size: 0.8rem;
    }
    
    .news-title {
        color: #333;
        font-weight: 400;
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
@st.cache_data(ttl=7200)  # Cache for 2 hours
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


@st.cache_data(ttl=7200)
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


@st.cache_data(ttl=7200)
def load_all_prices() -> pd.DataFrame:
    """Load all prices for all tickers."""
    rows = PriceRepository.get_all_prices_with_ticker()
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    return df


@st.cache_data(ttl=7200)
def load_sentiments() -> pd.DataFrame:
    """Load all sentiment scores."""
    scores = SentimentRepository.get_all_scores()
    if not scores:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {"date": s.date, "sentiment": float(s.score), "headlines": s.headlines_count, "explanation": s.explanation}
        for s in scores
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=7200)
def load_market_sentiment() -> dict:
    """Calculate overall market sentiment with AI explanation based on actual news."""
    from anime_stock.analysis.sentiment import SentimentAnalyzer
    
    # Get latest sentiment for each ticker
    latest_sentiments = {}
    explanations = {}
    company_headlines = {}
    
    # Get all tracked tickers
    stocks = load_tickers()
    for stock in stocks:
        ticker_scores = SentimentRepository.get_scores_for_ticker(stock["symbol"])
        if ticker_scores:
            latest = ticker_scores[-1]  # Most recent score
            latest_sentiments[stock["symbol"]] = float(latest.score)
            explanations[stock["symbol"]] = latest.explanation
            
            # Get latest headlines for this ticker (for context)
            ticker_news = load_news_for_ticker(stock["symbol"], limit=5)
            if ticker_news:
                company_headlines[stock["symbol"]] = [article["title"] for article in ticker_news]
    
    if not latest_sentiments:
        return {
            "score": 0.0,
            "explanation": "Немає даних для аналізу ринкового настрою",
            "tickers_analyzed": 0,
            "positive_count": 0,
            "negative_count": 0,
            "companies_with_news": 0
        }
    
    # Calculate market averages
    scores = list(latest_sentiments.values())
    avg_score = sum(scores) / len(scores)
    positive_count = sum(1 for s in scores if s > 0.05)
    negative_count = sum(1 for s in scores if s < -0.05)
    
    # Generate market explanation using AI with actual headlines
    try:
        analyzer = SentimentAnalyzer()
        explanation = analyzer.generate_market_explanation(
            avg_score, len(scores), positive_count, negative_count, company_headlines
        )
    except Exception as e:
        logger.error(f"Failed to generate market explanation: {e}")
        if avg_score > 0.3:
            explanation = "Загалом позитивний настрій на аніме та геймінг ринку"
        elif avg_score < -0.3:
            explanation = "Негативні тенденції в аніме та геймінг індустрії"
        else:
            explanation = "Змішаний настрій на ринку аніме та геймінгу"
    
    return {
        "score": avg_score,
        "explanation": explanation,
        "tickers_analyzed": len(scores),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "companies_with_news": len(company_headlines)
    }


@st.cache_data(ttl=7200)
def load_sentiments_for_ticker(ticker_symbol: str) -> pd.DataFrame:
    """Load sentiment scores for a specific ticker."""
    scores = SentimentRepository.get_scores_for_ticker(ticker_symbol)
    if not scores:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {"date": s.date, "sentiment": float(s.score), "headlines": s.headlines_count, "explanation": s.explanation}
        for s in scores
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=7200)
def load_latest_news(limit: int = 50) -> list[dict]:
    """Load latest news articles for tracked stocks, balanced across tickers."""
    # Request many more articles to ensure good distribution
    articles = NewsRepository.get_latest_articles(200)
    
    # Get tracked ticker symbols
    tickers = TickerRepository.get_all_active()
    tracked_symbols = {t.symbol for t in tickers}
    
    # Group articles by ticker
    from collections import defaultdict
    by_ticker = defaultdict(list)
    
    for a in articles:
        if a.ticker and a.ticker in tracked_symbols:
            by_ticker[a.ticker].append({
                "source": a.source,
                "title": a.title,
                "title_uk": a.title_uk,
                "url": a.url,
                "published_at": a.published_at,
                "ticker": a.ticker,
            })
    
    # Distribute fairly: take 1-3 articles from each ticker in round-robin
    balanced = []
    max_per_ticker = 3  # Maximum articles per ticker in the feed
    
    for ticker in sorted(by_ticker.keys()):
        balanced.extend(by_ticker[ticker][:max_per_ticker])
    
    # Sort by published date (newest first) and return top 10
    balanced.sort(key=lambda x: x["published_at"] or "", reverse=True)
    return balanced[:10]


@st.cache_data(ttl=7200)
def load_news_for_ticker(ticker_symbol: str, limit: int = 10) -> list[dict]:
    """Load latest news articles for a specific ticker."""
    # Fetch more articles to ensure we get enough for this ticker
    articles = NewsRepository.get_latest_articles(500)  # Increased from 30 to 500
    
    filtered = [
        {
            "source": a.source,
            "title": a.title,
            "title_uk": a.title_uk,
            "url": a.url,
            "published_at": a.published_at,
            "ticker": a.ticker,
        }
        for a in articles
        if a.ticker == ticker_symbol
    ]
    
    return filtered[:limit]


@st.cache_data(ttl=7200)
def load_predictions_for_ticker(ticker_id: int) -> pd.DataFrame:
    """Load all predictions for a specific ticker (both verified and future)."""
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
            WHERE ticker_id = %s
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
def get_exchange_rate(from_currency: str = "USD", to_currency: str = "JPY") -> float:
    """Get exchange rate between two currencies."""
    try:
        # Try database first
        cached = ExchangeRateRepository.get_latest_rate(from_currency, to_currency)
        if cached:
            return cached
        
        # Fetch from API if not in database
        response = requests.get(
            f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}",
            timeout=5
        )
        if response.ok:
            rate = response.json()["rates"][to_currency]
            ExchangeRateRepository.insert_rate(from_currency, to_currency, rate, date.today())
            return rate
    except Exception:
        pass
    
    # Fallback defaults
    if from_currency == "USD" and to_currency == "JPY":
        return 150.0
    elif from_currency == "USD" and to_currency == "UAH":
        return 40.0
    return 1.0


def convert_price(price: float, from_currency: str, to_currency: str, usd_jpy_rate: float, usd_uah_rate: float = 40.0) -> float:
    """Convert price between currencies (USD, JPY, UAH)."""
    if from_currency == to_currency:
        return price
    
    # Convert to USD first as intermediary
    if from_currency == "JPY":
        price_usd = price / usd_jpy_rate
    elif from_currency == "UAH":
        price_usd = price / usd_uah_rate
    else:  # from_currency == "USD"
        price_usd = price
    
    # Convert from USD to target currency
    if to_currency == "JPY":
        return price_usd * usd_jpy_rate
    elif to_currency == "UAH":
        return price_usd * usd_uah_rate
    else:  # to_currency == "USD"
        return price_usd


@st.cache_data(ttl=7200)
def calculate_anime_index(display_currency: str, usd_jpy_rate: float, usd_uah_rate: float) -> pd.DataFrame:
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
                lambda x: convert_price(x, orig_currency, display_currency, usd_jpy_rate, usd_uah_rate) if pd.notna(x) else x
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
            
            # Create hover text with headlines count
            hover_texts = [
                f"<b>Sentiment: {score:.2f}</b><br>Based on {int(count)} news headlines<br>{'Positive 📈' if score >= 0 else 'Negative 📉'}"
                for score, count in zip(sent_filtered["sentiment"], sent_filtered["headlines"])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=sent_filtered.index,
                    y=sent_filtered["sentiment"],
                    name=get_text("sentiment", lang),
                    marker_color=colors,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts,
                ),
                row=2, col=1,
            )
    
    # Layout - Clean light theme
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=500,
        margin=dict(t=80, b=20, l=50, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(color="#333", size=11),
        hovermode="x unified",
        dragmode=False,
    )
    
    fig.update_yaxes(title_text=get_text("index_value", lang), row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text=get_text("score", lang), row=2, col=1, range=[-1.1, 1.1], gridcolor="#f0f0f0")
    fig.update_xaxes(gridcolor="#f0f0f0")
    
    return fig


def create_price_chart(
    df: pd.DataFrame,
    sentiments: pd.DataFrame,
    predictions: pd.DataFrame,
    ticker_name: str,
    display_currency: str,
    ticker_currency: str,
    usd_jpy_rate: float,
    usd_uah_rate: float,
    lang: str = "en",
) -> go.Figure:
    """Create the price chart for a single stock with AI predictions."""
    
    # Convert prices if needed
    df = df.copy()
    if ticker_currency != display_currency:
        for col in ["open", "high", "low", "close", "sma_20", "sma_50"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: convert_price(x, ticker_currency, display_currency, usd_jpy_rate, usd_uah_rate) if pd.notna(x) else x
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
        # Don't filter by price date range - show all predictions including future
        pred_filtered = predictions.copy()
        
        if not pred_filtered.empty:
            # Get actual prices for prediction dates (use left join to keep predictions without prices)
            pred_with_price = pred_filtered.join(df[["close"]], how="left")
            
            # For predictions without price data yet, use the last known price
            if pred_with_price["close"].isna().any():
                last_price = df["close"].iloc[-1]
                pred_with_price["close"] = pred_with_price["close"].fillna(last_price)
            
            # Separate correct and incorrect predictions
            up_correct = pred_with_price[(pred_with_price["direction"] == "UP") & (pred_with_price["actual_direction"] == "UP")]
            up_wrong = pred_with_price[(pred_with_price["direction"] == "UP") & (pred_with_price["actual_direction"] == "DOWN")]
            down_correct = pred_with_price[(pred_with_price["direction"] == "DOWN") & (pred_with_price["actual_direction"] == "DOWN")]
            down_wrong = pred_with_price[(pred_with_price["direction"] == "DOWN") & (pred_with_price["actual_direction"] == "UP")]
            
            # Future predictions (not yet verified)
            up_future = pred_with_price[(pred_with_price["direction"] == "UP") & (pred_with_price["actual_direction"].isna())]
            down_future = pred_with_price[(pred_with_price["direction"] == "DOWN") & (pred_with_price["actual_direction"].isna())]
            
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
                )
            
            # UP predictions - future (blue triangle up)
            if not up_future.empty:
                fig.add_trace(
                    go.Scatter(
                        x=up_future.index,
                        y=up_future["close"],
                        mode="markers",
                        name="🔮 AI: UP (Forecast)",
                        marker=dict(symbol="triangle-up", size=12, color="#007bff", line=dict(width=2, color="white")),
                        hovertemplate="<b>AI Predicts: UP</b><br>Pending verification 🔮<br>Confidence: %{customdata:.0%}<extra></extra>",
                        customdata=up_future["confidence"],
                    ),
                    row=1, col=1,
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
            
            # DOWN predictions - future (orange triangle down)
            if not down_future.empty:
                fig.add_trace(
                    go.Scatter(
                        x=down_future.index,
                        y=down_future["close"],
                        mode="markers",
                        name="🔮 AI: DOWN (Forecast)",
                        marker=dict(symbol="triangle-down", size=12, color="#fd7e14", line=dict(width=2, color="white")),
                        hovertemplate="<b>AI Predicts: DOWN</b><br>Pending verification 🔮<br>Confidence: %{customdata:.0%}<extra></extra>",
                        customdata=down_future["confidence"],
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
            
            # Create hover text with headlines count
            hover_texts = [
                f"<b>Sentiment: {score:.2f}</b><br>Based on {int(count)} news headlines<br>{'Positive 📈' if score >= 0 else 'Negative 📉'}"
                for score, count in zip(sent_filtered["sentiment"], sent_filtered["headlines"])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=sent_filtered.index,
                    y=sent_filtered["sentiment"],
                    name=get_text("sentiment", lang),
                    marker_color=colors,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts,
                ),
                row=3, col=1,
            )
    
    # Layout - Light theme
    if display_currency == "JPY":
        currency_symbol = "¥"
    elif display_currency == "UAH":
        currency_symbol = "₴"
    else:
        currency_symbol = "$"
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=600,
        margin=dict(t=80, b=20, l=50, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(color="#333", size=11),
        hovermode="x unified",
        dragmode=False,
    )
    
    fig.update_yaxes(title_text=f"{get_text('price', lang)} ({currency_symbol})", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text=get_text("volume", lang), row=2, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text=get_text("score", lang), row=3, col=1, range=[-1.1, 1.1], gridcolor="#f0f0f0")
    fig.update_xaxes(gridcolor="#f0f0f0")
    
    return fig


# --- MAIN APP ---
def main():
    """Main Streamlit application."""
    
    # Apply custom CSS
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    # Initialize language and page in session state (use language-independent keys)
    if "lang" not in st.session_state:
        st.session_state.lang = "uk"
    if "page" not in st.session_state:
        st.session_state.page = "Index"  # Store language-independent value
    
    # --- HEADER WITH BRANDING ---
    st.markdown(
        f"""
        <div style='background: #ffffff; padding: 12px 20px; border-bottom: 1px solid #e0e0e0; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h1 style='color: #333; margin: 0; font-size: 20px; font-weight: 400; letter-spacing: 0.5px;'>Аніме-аналітика</h1>
                    <p style='color: #999; margin: 4px 0 0 0; font-size: 10px; font-style: italic;'>Fun project · Not financial advice</p>
                </div>
                <div style='text-align: right; color: #666; font-size: 14px; font-weight: 500;'>
                    {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load data
    tickers = load_tickers()
    
    if not tickers:
        st.warning(get_text("no_tickers", st.session_state.lang))
        return
    
    # Compact control bar - single row with dropdowns
    rate_usd_jpy = get_exchange_rate("USD", "JPY")
    rate_usd_uah = get_exchange_rate("USD", "UAH")
    sentiments_df = load_sentiments()
    
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 3, 1])
    with c1:
        display_currency = st.selectbox(get_text("currency", st.session_state.lang), ["USD", "JPY", "UAH"], key="currency", label_visibility="collapsed")
    
    # Set currency symbol based on selected currency
    if display_currency == "JPY":
        currency_symbol = "¥"
    elif display_currency == "UAH":
        currency_symbol = "₴"
    else:
        currency_symbol = "$"
    
    with c2:
        # Build page options with current language
        page_options = [get_text("view_index", st.session_state.lang), get_text("view_stocks", st.session_state.lang)]
        page_index = 0 if st.session_state.page == "Index" else 1
        
        selected_page_text = st.selectbox(
            get_text("view", st.session_state.lang),
            page_options,
            index=page_index,
            key="page_selector",
            label_visibility="collapsed"
        )
        
        # Convert selected text back to language-independent value
        st.session_state.page = "Index" if selected_page_text == get_text("view_index", st.session_state.lang) else "Stocks"
    with c3:
        date_range = st.selectbox(get_text("period", st.session_state.lang), ["1M", "3M", "6M", "1Y", "2Y"], index=3, label_visibility="collapsed")
    with c4:
        # Calculate JPY/UAH rate (how many UAH per 1 JPY)
        rate_jpy_uah = rate_usd_uah / rate_usd_jpy if rate_usd_jpy > 0 else 0
        
        # Display exchange rates only (no duplication)
        st.markdown(
            f"""
            <div style='padding: 6px 12px; background: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6; font-size: 11px;'>
                <div style='display: flex; justify-content: space-around; align-items: center; color: #495057;'>
                    <span>USD/JPY: <b style='color: #28a745;'>¥{rate_usd_jpy:.2f}</b></span>
                    <span>USD/UAH: <b style='color: #28a745;'>₴{rate_usd_uah:.2f}</b></span>
                    <span>JPY/UAH: <b style='color: #28a745;'>₴{rate_jpy_uah:.4f}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c5:
        lang_option = st.selectbox(
            "Language",
            options=["🇺🇦 UA", "🇺🇸 EN"],
            index=0 if st.session_state.lang == "uk" else 1,
            key="language",
            label_visibility="collapsed"
        )
        st.session_state.lang = "uk" if "UA" in lang_option else "en"
    
    # Use the language-independent page value
    page_normalized = st.session_state.page
    
    if page_normalized == "Index":
        # --- ANIME INDEX PAGE ---
        
        # Calculate index
        index_df = calculate_anime_index(display_currency, rate_usd_jpy, rate_usd_uah)
        
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
            # Clean light metric card
            change_color = "#28a745" if index_change_pct >= 0 else "#dc3545"
            st.markdown(
                f"""
                <div style='background: #ffffff !important; padding: 16px; border-radius: 6px; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div style='color: #6c757d; font-size: 12px; margin-bottom: 8px;'>{get_text("anime_index", st.session_state.lang)}</div>
                    <div style='font-size: 28px; font-weight: 600; color: #212529; margin-bottom: 4px;'>{current_index:.1f}</div>
                    <div style='color: {change_color}; font-size: 14px; font-weight: 500;'>{"↗" if index_change_pct >= 0 else "↘"} {index_change_pct:+.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Load market sentiment with AI explanation
        market_mood = load_market_sentiment()
        latest_sentiment = market_mood["score"]
        sentiment_explanation = market_mood["explanation"]
        
        sentiment_label = (
            get_text("sentiment_bullish", st.session_state.lang) if latest_sentiment > 0.3
            else get_text("sentiment_bearish", st.session_state.lang) if latest_sentiment < -0.3
            else get_text("sentiment_neutral", st.session_state.lang)
        )
        
        with col2:
            # Market sentiment card - compact version without explanation
            sent_color = "#28a745" if latest_sentiment > 0.3 else "#dc3545" if latest_sentiment < -0.3 else "#ffc107"
            sent_emoji = "😊" if latest_sentiment > 0.3 else "😟" if latest_sentiment < -0.3 else "😐"
            
            st.markdown(
                f"""
                <div style='background: #ffffff; padding: 16px; border-radius: 6px; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div style='color: #6c757d; font-size: 12px; margin-bottom: 8px;'>{get_text("market_sentiment", st.session_state.lang)}</div>
                    <div style='font-size: 28px; font-weight: 600; color: #212529; margin-bottom: 4px;'>{latest_sentiment:.2f}</div>
                    <div style='color: {sent_color}; font-size: 14px;'>{sent_emoji} {sentiment_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Exchange rate
        with col3:
            st.markdown(
                f"""
                <div style='background: #ffffff; padding: 16px; border-radius: 6px; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div style='color: #6c757d; font-size: 12px; margin-bottom: 8px;'>{get_text("usd_jpy", st.session_state.lang)}</div>
                    <div style='font-size: 28px; font-weight: 600; color: #212529; margin-bottom: 4px;'>¥{rate_usd_jpy:.2f}</div>
                    <div style='color: #6c757d; font-size: 13px;'>🔴 {get_text("live", st.session_state.lang)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Tracked stocks count
        with col4:
            st.markdown(
                f"""
                <div style='background: #ffffff; padding: 16px; border-radius: 6px; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div style='color: #6c757d; font-size: 12px; margin-bottom: 8px;'>{get_text("tracked_stocks", st.session_state.lang)}</div>
                    <div style='font-size: 28px; font-weight: 600; color: #212529; margin-bottom: 4px;'>{len(tickers)}</div>
                    <div style='color: #6c757d; font-size: 13px;'>✓ {get_text("active", st.session_state.lang)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # --- INFO BOX: What is Anime Index ---
        st.markdown(
            f"""
            <div style='padding: 12px 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #667eea; border-radius: 6px; margin-bottom: 16px;'>
                <div style='display: flex; align-items: flex-start;'>
                    <span style='font-size: 20px; margin-right: 10px; margin-top: 2px;'>�</span>
                    <div>
                        <h4 style='margin: 0 0 6px 0; color: #667eea; font-size: 16px;'>{get_text("info_index_title", st.session_state.lang)}</h4>
                        <p style='margin: 0; color: #555; font-size: 13px; line-height: 1.5;'>{get_text("info_index_text", st.session_state.lang)}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- MAIN CHART AND NEWS SIDE BY SIDE ---
        chart_col, news_col = st.columns([2, 1])
        
        with chart_col:
            chart = create_index_chart(index_df, sentiments_df, st.session_state.lang)
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
            
            # Market explanation under chart
            sent_color = "#28a745" if latest_sentiment > 0.3 else "#dc3545" if latest_sentiment < -0.3 else "#ffc107"
            sent_emoji = "😊" if latest_sentiment > 0.3 else "😟" if latest_sentiment < -0.3 else "😐"
            
            st.markdown(
                f"""
                <div style='padding: 16px; background: linear-gradient(135deg, {sent_color}15 0%, {sent_color}10 100%); border-left: 4px solid {sent_color}; border-radius: 8px; margin-top: 16px;'>
                    <div style='display: flex; align-items: flex-start;'>
                        <span style='font-size: 20px; margin-right: 12px; margin-top: 2px;'>{sent_emoji}</span>
                        <div>
                            <h4 style='margin: 0 0 8px 0; color: {sent_color}; font-size: 16px;'>{get_text("market_sentiment", st.session_state.lang)}: {latest_sentiment:.2f}</h4>
                            <p style='margin: 0 0 12px 0; color: #333; font-size: 14px; line-height: 1.5;'>{sentiment_explanation}</p>
                            <div style='color: #666; font-size: 12px; border-top: 1px solid #e9ecef; padding-top: 8px;'>
                                📊 {market_mood["tickers_analyzed"]} companies | 📰 {market_mood["companies_with_news"]} with news | 📈 {market_mood["positive_count"]} positive | 📉 {market_mood["negative_count"]} negative
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with news_col:
            st.subheader(get_text("latest_news", st.session_state.lang))
            news_items = load_latest_news(10)
            
            if news_items:
                for item in news_items:
                    # Display ticker symbol prominently
                    ticker_tag = f"{item.get('ticker', '???')}" if item.get('ticker') else ""
                    pub_date = format_date(item["published_at"], st.session_state.lang, "short") if item["published_at"] else ""
                    
                    # Use Ukrainian title if language is Ukrainian and translation exists
                    if st.session_state.lang == "uk" and item.get('title_uk'):
                        title = item['title_uk']
                    else:
                        title = item['title']
                    
                    st.markdown(
                        f"""
                        <div style='background: white; padding: 14px 16px; border-radius: 8px; margin-bottom: 12px; 
                                    border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.08); 
                                    transition: box-shadow 0.3s ease; cursor: pointer;'
                             onmouseover="this.style.boxShadow='0 4px 12px rgba(102,126,234,0.15)'"
                             onmouseout="this.style.boxShadow='0 2px 4px rgba(0,0,0,0.08)'">
                            <div style='color: #667eea; font-size: 12px; font-weight: 600; margin-bottom: 6px;'>
                                📈 {ticker_tag} • {item['source']} • {pub_date}
                            </div>
                            <div style='color: #333; font-size: 14px; line-height: 1.4;'>
                                <a href="{item['url']}" target="_blank" style='color: #333; text-decoration: none;'>
                                    {title[:100]}{'...' if len(title) > 100 else ''}
                                </a>
                            </div>
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
        
        # Ticker selector at the top
        ticker_options = {f"{t['symbol']} - {t['name']}": t for t in tickers}
        selected = st.selectbox(
            get_text("select_stock", st.session_state.lang),
            options=list(ticker_options.keys()),
        )
        selected_ticker = ticker_options[selected]
        
        # Stock-specific info box
        st.markdown(
            f"""
            <div style='padding: 12px 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #764ba2; border-radius: 6px; margin-bottom: 16px;'>
                <div style='display: flex; align-items: flex-start;'>
                    <span style='font-size: 20px; margin-right: 10px; margin-top: 2px;'>📈</span>
                    <div>
                        <h4 style='margin: 0 0 6px 0; color: #764ba2; font-size: 16px;'>{selected_ticker['name']} ({selected_ticker['symbol']})</h4>
                        <p style='margin: 0; color: #555; font-size: 13px; line-height: 1.5;'>{get_text("sector", st.session_state.lang)}: {selected_ticker['sector'].capitalize()} • {get_text("currency", st.session_state.lang)}: {selected_ticker['currency']}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Current price + prediction card + 52-week range
        col0, col1, col2, col3 = st.columns([1, 1, 1, 1])
        
        # Load price data and ticker-specific sentiment
        prices_df = load_prices(selected_ticker["id"])
        ticker_sentiments_df = load_sentiments_for_ticker(selected_ticker["symbol"])
        
        # Load market sentiment for the compact card and detailed explanation
        market_mood = load_market_sentiment()
        latest_sentiment = market_mood["score"]
        
        # Fallback to global sentiment if no ticker-specific data
        if ticker_sentiments_df.empty:
            ticker_sentiments_df = load_sentiments()  # Fallback to global sentiment
        
        # Current stock price badge
        with col0:
            if not prices_df.empty:
                current_price = prices_df["close"].iloc[-1]
                prev_price = prices_df["close"].iloc[-2] if len(prices_df) > 1 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                change_color = "#28a745" if price_change_pct >= 0 else "#dc3545"
                
                # Convert price to display currency
                display_price = convert_price(current_price, selected_ticker["currency"], display_currency, rate_usd_jpy, rate_usd_uah)
                
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #667eea15, #764ba215); padding: 16px; border-radius: 8px; border-left: 4px solid #667eea;'>
                        <div style='color: #666; font-size: 13px; margin-bottom: 8px;'>{get_text("current_price", st.session_state.lang)}</div>
                        <div style='font-size: 24px; font-weight: bold; color: #333; margin-bottom: 4px;'>{currency_symbol}{display_price:,.2f}</div>
                        <div style='color: {change_color}; font-size: 14px;'>{"+" if price_change_pct >= 0 else ""}{price_change_pct:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col1:
            # Compact sentiment card for stocks page
            sent_color = "#28a745" if latest_sentiment > 0.3 else "#dc3545" if latest_sentiment < -0.3 else "#ffc107"
            sent_emoji = "😊" if latest_sentiment > 0.3 else "😟" if latest_sentiment < -0.3 else "😐"
            sent_label = (
                get_text("sentiment_bullish", st.session_state.lang) if latest_sentiment > 0.3
                else get_text("sentiment_bearish", st.session_state.lang) if latest_sentiment < -0.3
                else get_text("sentiment_neutral", st.session_state.lang)
            )
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea15, #764ba215); padding: 16px; border-radius: 8px; border-left: 4px solid {sent_color};'>
                    <div style='color: #666; font-size: 13px; margin-bottom: 8px;'>{get_text("market_sentiment", st.session_state.lang)}</div>
                    <div style='font-size: 32px; font-weight: bold; color: #333; margin-bottom: 4px;'>{latest_sentiment:.2f}</div>
                    <div style='color: {sent_color}; font-size: 14px;'>{sent_emoji} {sent_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Prediction
        prediction = PredictionRepository.get_latest_prediction(selected_ticker["id"])
        if prediction:
            pred_direction = prediction.direction
            pred_confidence = float(prediction.confidence)
            pred_emoji = "📈" if pred_direction == "UP" else "📉"
            pred_color = "#28a745" if pred_direction == "UP" else "#dc3545"
        else:
            pred_direction = "N/A"
            pred_confidence = 0.0
            pred_emoji = "❓"
            pred_color = "#999"
        
        with col2:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea15, #764ba215); padding: 16px; border-radius: 8px; border-left: 4px solid {pred_color};'>
                    <div style='color: #666; font-size: 13px; margin-bottom: 8px;'>{get_text("ai_forecast", st.session_state.lang)}</div>
                    <div style='font-size: 32px; font-weight: bold; color: #333; margin-bottom: 4px;'>{pred_emoji} {pred_direction}</div>
                    <div style='color: {pred_color}; font-size: 14px;'>{f"{pred_confidence:.0%} {get_text('confidence', st.session_state.lang)}" if prediction else get_text("no_prediction", st.session_state.lang)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # 52-week range
        year_prices = prices_df["close"].tail(252)
        with col3:
            high_price = convert_price(year_prices.max(), selected_ticker["currency"], display_currency, rate_usd_jpy, rate_usd_uah)
            low_price = convert_price(year_prices.min(), selected_ticker["currency"], display_currency, rate_usd_jpy, rate_usd_uah)
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea15, #764ba215); padding: 16px; border-radius: 8px; border-left: 4px solid #764ba2;'>
                    <div style='color: #666; font-size: 13px; margin-bottom: 8px;'>{get_text("week_range", st.session_state.lang)}</div>
                    <div style='font-size: 20px; font-weight: bold; color: #333; margin-bottom: 4px;'>{currency_symbol}{low_price:,.0f} - {currency_symbol}{high_price:,.0f}</div>
                    <div style='color: #764ba2; font-size: 14px;'>52 weeks</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # --- MAIN CHART ---
        st.subheader(f"📊 {selected_ticker['name']} ({selected_ticker['symbol']})")
        
        # Load predictions for this ticker
        predictions_df = load_predictions_for_ticker(selected_ticker["id"])
        
        # Debug: show prediction stats
        if not predictions_df.empty:
            verified = predictions_df[predictions_df["actual_direction"].notna()].shape[0]
            future = predictions_df[predictions_df["actual_direction"].isna()].shape[0]
            st.info(f"📈 Predictions: {verified} verified, {future} future forecasts")
        else:
            st.warning(f"⚠️ No predictions found for {selected_ticker['symbol']}")
        
        chart = create_price_chart(
            prices_df,
            ticker_sentiments_df,  # Use ticker-specific sentiment
            predictions_df,
            selected_ticker["name"],
            display_currency,
            selected_ticker["currency"],
            rate_usd_jpy,
            rate_usd_uah,
            st.session_state.lang,
        )
        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
        
        # --- SENTIMENT EXPLANATION BOX (right after chart) ---
        if not ticker_sentiments_df.empty:
            latest_sent_score = ticker_sentiments_df.iloc[-1]["sentiment"]
            latest_explanation = ticker_sentiments_df.iloc[-1]["explanation"] if "explanation" in ticker_sentiments_df.columns else None
            
            if latest_explanation:
                # Determine color based on sentiment
                if latest_sent_score > 0.05:
                    exp_color = "#28a745"
                    exp_emoji = "📈"
                elif latest_sent_score < -0.05:
                    exp_color = "#dc3545"
                    exp_emoji = "📉"
                else:
                    exp_color = "#6c757d"
                    exp_emoji = "😐"
                
                st.markdown(
                    f"""
                    <div style='padding: 16px 20px; background: {exp_color}10; border-left: 5px solid {exp_color}; border-radius: 6px; margin: 20px 0;'>
                        <div style='color: #666; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>
                            AI Sentiment Analysis
                        </div>
                        <div style='color: #333; font-size: 16px; line-height: 1.6; font-weight: 500;'>
                            {exp_emoji} {latest_explanation}
                        </div>
                        <div style='color: #999; font-size: 11px; margin-top: 8px;'>
                            Score: {latest_sent_score:.2f} | Based on recent news headlines
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # --- NEWS SECTION: Headlines used for sentiment ---
        st.markdown("---")
        st.subheader(get_text("news_used_for_sentiment", st.session_state.lang))
        
        # Get latest sentiment score and explanation for this ticker
        latest_sentiment = None
        sentiment_explanation = None
        if not ticker_sentiments_df.empty:
            latest_sentiment = ticker_sentiments_df.iloc[-1]["sentiment"]
            sentiment_explanation = ticker_sentiments_df.iloc[-1]["explanation"] if "explanation" in ticker_sentiments_df.columns else None
        
        # Load news articles for this ticker
        ticker_news = load_news_for_ticker(selected_ticker["symbol"], limit=10)
        
        if ticker_news:
            # Show sentiment result if available
            if latest_sentiment is not None:
                # Determine sentiment color, emoji and text
                if latest_sentiment > 0.05:  # Positive threshold
                    sentiment_color = "#28a745"
                    sentiment_emoji = "📈"
                    sentiment_text = "Positive"
                elif latest_sentiment < -0.05:  # Negative threshold
                    sentiment_color = "#dc3545"
                    sentiment_emoji = "📉"
                    sentiment_text = "Negative"
                else:  # Neutral (-0.05 to 0.05)
                    sentiment_color = "#6c757d"
                    sentiment_emoji = "😐"
                    sentiment_text = "Neutral"
                
                # Use AI explanation if available
                display_explanation = sentiment_explanation if sentiment_explanation else ""
                
                st.markdown(
                    f"""
                    <div style='padding: 12px 16px; background: {sentiment_color}15; border-left: 4px solid {sentiment_color}; border-radius: 6px; margin-bottom: 16px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: #666; font-size: 13px;'>Latest Sentiment Score:</span>
                                <span style='color: {sentiment_color}; font-size: 20px; font-weight: bold; margin-left: 10px;'>{sentiment_emoji} {latest_sentiment:.2f}</span>
                                <span style='color: #666; font-size: 13px; margin-left: 10px;'>({sentiment_text})</span>
                            </div>
                            <div style='color: #999; font-size: 12px;'>Based on {len(ticker_news)} articles</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Sentiment is 0 or not calculated yet
                st.info(f"ℹ️ Sentiment analysis not yet calculated for these articles. Run the daily collection script to analyze.")
            
            st.markdown(f"<p style='color: #666; font-size: 14px; margin-bottom: 12px;'>{get_text('recent_headlines', st.session_state.lang)}</p>", unsafe_allow_html=True)
            
            # Display news articles
            for i, article in enumerate(ticker_news, 1):
                title = article["title_uk"] if st.session_state.lang == "uk" and article["title_uk"] else article["title"]
                published = article["published_at"].strftime("%d %b %Y") if article["published_at"] else "Unknown date"
                
                st.markdown(
                    f"""
                    <div style='padding: 10px 14px; background: #f8f9fa; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #007bff;'>
                        <div style='display: flex; justify-content: space-between; align-items: start;'>
                            <div style='flex: 1;'>
                                <span style='color: #007bff; font-weight: 500; font-size: 12px; margin-right: 8px;'>#{i}</span>
                                <a href='{article["url"]}' target='_blank' style='color: #333; text-decoration: none; font-size: 14px; line-height: 1.4;'>{title}</a>
                            </div>
                            <span style='color: #999; font-size: 11px; white-space: nowrap; margin-left: 12px;'>{published}</span>
                        </div>
                        <div style='color: #666; font-size: 11px; margin-top: 4px; margin-left: 24px;'>{article["source"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info(get_text("no_news_ticker", st.session_state.lang))
    
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
