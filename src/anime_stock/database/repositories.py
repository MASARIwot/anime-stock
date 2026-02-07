"""Database repositories for Animetrics AI."""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from anime_stock.database.connection import get_connection

logger = logging.getLogger(__name__)


@dataclass
class Ticker:
    """Stock ticker entity."""

    id: int
    symbol: str
    company_name: str
    exchange: str
    sector: str
    currency: str
    active: bool


@dataclass
class StockPrice:
    """Stock price entity."""

    id: int
    ticker_id: int
    date: date
    open: Optional[Decimal]
    high: Optional[Decimal]
    low: Optional[Decimal]
    close: Decimal
    volume: Optional[int]
    created_at: Optional[datetime] = None


@dataclass
class NewsArticle:
    """News article entity."""

    id: int
    source: str
    title: str
    url: str
    published_at: Optional[datetime]
    scraped_at: datetime
    ticker: Optional[str] = None
    title_uk: Optional[str] = None  # Ukrainian translation


@dataclass
class SentimentScore:
    """Sentiment score entity."""

    id: int
    date: date
    score: Decimal
    model_used: str
    headlines_count: int
    ticker: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class Prediction:
    """Prediction entity."""

    id: int
    ticker_id: int
    date: date
    direction: str
    confidence: Decimal
    model_version: str
    actual_direction: Optional[str]


class TickerRepository:
    """Repository for stock tickers."""

    @staticmethod
    def get_all_active() -> list[Ticker]:
        """Get all active tickers."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, symbol, company_name, exchange, sector, currency, active "
                "FROM stock_tickers WHERE active = 1"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [Ticker(**row) for row in rows]

    @staticmethod
    def get_by_symbol(symbol: str) -> Optional[Ticker]:
        """Get ticker by symbol."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, symbol, company_name, exchange, sector, currency, active "
                "FROM stock_tickers WHERE symbol = %s",
                (symbol,),
            )
            row = cursor.fetchone()
            cursor.close()
            return Ticker(**row) if row else None

    @staticmethod
    def get_by_id(ticker_id: int) -> Optional[Ticker]:
        """Get ticker by ID."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, symbol, company_name, exchange, sector, currency, active "
                "FROM stock_tickers WHERE id = %s",
                (ticker_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            return Ticker(**row) if row else None


class PriceRepository:
    """Repository for stock prices."""

    @staticmethod
    def get_last_date(ticker_id: int) -> Optional[date]:
        """Get the most recent date for a ticker."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(date) FROM stock_prices WHERE ticker_id = %s",
                (ticker_id,),
            )
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result and result[0] else None

    @staticmethod
    def insert_prices(prices: list[dict]) -> int:
        """Insert multiple price records. Returns number of inserted rows."""
        if not prices:
            return 0

        with get_connection() as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO stock_prices (ticker_id, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    open = VALUES(open),
                    high = VALUES(high),
                    low = VALUES(low),
                    close = VALUES(close),
                    volume = VALUES(volume)
            """
            params = [
                (
                    p["ticker_id"],
                    p["date"],
                    p.get("open"),
                    p.get("high"),
                    p.get("low"),
                    p["close"],
                    p.get("volume"),
                )
                for p in prices
            ]
            cursor.executemany(query, params)
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            logger.info(f"Inserted/updated {affected} price records")
            return affected

    @staticmethod
    def get_prices_for_ticker(
        ticker_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[StockPrice]:
        """Get prices for a ticker within date range."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM stock_prices WHERE ticker_id = %s"
            params = [ticker_id]

            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)

            query += " ORDER BY date ASC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            cursor.close()
            return [StockPrice(**row) for row in rows]

    @staticmethod
    def get_all_prices_with_ticker() -> list[dict]:
        """Get all prices joined with ticker info."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT sp.*, st.symbol, st.company_name, st.currency
                FROM stock_prices sp
                JOIN stock_tickers st ON sp.ticker_id = st.id
                ORDER BY sp.date ASC
            """)
            rows = cursor.fetchall()
            cursor.close()
            return rows


class NewsRepository:
    """Repository for news articles."""

    @staticmethod
    def insert_articles(articles: list[dict]) -> int:
        """Insert news articles, ignoring duplicates by URL hash."""
        if not articles:
            return 0

        with get_connection() as conn:
            cursor = conn.cursor()
            query = """
                INSERT IGNORE INTO news_articles (source, title, title_uk, url, published_at, ticker)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = [
                (a["source"], a["title"], a.get("title_uk"), a["url"], a.get("published_at"), a.get("ticker"))
                for a in articles
            ]
            cursor.executemany(query, params)
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            logger.info(f"Inserted {affected} new articles")
            return affected

    @staticmethod
    def get_articles_for_date(target_date: date, ticker: Optional[str] = None) -> list[NewsArticle]:
        """Get articles published on a specific date, optionally filtered by ticker."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            if ticker:
                cursor.execute(
                    """
                    SELECT id, source, title, title_uk, url, published_at, scraped_at, ticker
                    FROM news_articles 
                    WHERE DATE(published_at) = %s AND ticker = %s
                    ORDER BY published_at ASC
                    """,
                    (target_date, ticker),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, source, title, title_uk, url, published_at, scraped_at, ticker
                    FROM news_articles 
                    WHERE DATE(published_at) = %s
                    ORDER BY published_at ASC
                    """,
                    (target_date,),
                )
            rows = cursor.fetchall()
            cursor.close()
            return [NewsArticle(**row) for row in rows]

    @staticmethod
    def get_unprocessed_dates() -> list[tuple[date, str]]:
        """Get (date, ticker) pairs with news articles but no sentiment score."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT DATE(na.published_at) as news_date, na.ticker
                FROM news_articles na
                LEFT JOIN sentiment_scores ss 
                    ON DATE(na.published_at) = ss.date AND na.ticker = ss.ticker
                WHERE ss.id IS NULL AND na.published_at IS NOT NULL AND na.ticker IS NOT NULL
                ORDER BY news_date ASC, na.ticker ASC
            """)
            rows = cursor.fetchall()
            cursor.close()
            return [(row[0], row[1]) for row in rows if row[0] and row[1]]

    @staticmethod
    def get_latest_articles(limit: int = 10) -> list[NewsArticle]:
        """Get the most recent news articles."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, source, title, title_uk, url, published_at, scraped_at, ticker
                FROM news_articles 
                WHERE published_at IS NOT NULL
                ORDER BY published_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [NewsArticle(**row) for row in rows]


class SentimentRepository:
    """Repository for sentiment scores."""

    @staticmethod
    def insert_score(
        target_date: date,
        score: float,
        model_used: str,
        headlines_count: int,
        ticker: str,
        raw_headlines: Optional[str] = None,
        explanation: Optional[str] = None,
    ) -> int:
        """Insert or update a sentiment score for a date and ticker."""
        with get_connection() as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO sentiment_scores (date, score, model_used, headlines_count, ticker, raw_headlines, explanation)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    score = VALUES(score),
                    model_used = VALUES(model_used),
                    headlines_count = VALUES(headlines_count),
                    raw_headlines = VALUES(raw_headlines),
                    explanation = VALUES(explanation),
                    updated_at = CURRENT_TIMESTAMP
            """
            cursor.execute(query, (target_date, score, model_used, headlines_count, ticker, raw_headlines, explanation))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            return affected

    @staticmethod
    def get_score_for_date(target_date: date, ticker: Optional[str] = None) -> Optional[SentimentScore]:
        """Get sentiment score for a specific date and optionally ticker."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            if ticker:
                cursor.execute(
                    "SELECT id, date, score, model_used, headlines_count, ticker "
                    "FROM sentiment_scores WHERE date = %s AND ticker = %s",
                    (target_date, ticker),
                )
            else:
                cursor.execute(
                    "SELECT id, date, score, model_used, headlines_count, ticker "
                    "FROM sentiment_scores WHERE date = %s",
                    (target_date,),
                )
            row = cursor.fetchone()
            cursor.close()
            return SentimentScore(**row) if row else None

    @staticmethod
    def get_all_scores() -> list[SentimentScore]:
        """Get all sentiment scores."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, date, score, model_used, headlines_count, ticker, explanation "
                "FROM sentiment_scores ORDER BY date ASC, ticker ASC"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [SentimentScore(**row) for row in rows]
    
    @staticmethod
    def get_scores_for_ticker(ticker_symbol: str) -> list[SentimentScore]:
        """Get all sentiment scores for a specific ticker."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, date, score, model_used, headlines_count, ticker, explanation "
                "FROM sentiment_scores WHERE ticker = %s ORDER BY date ASC",
                (ticker_symbol,)
            )
            rows = cursor.fetchall()
            cursor.close()
            return [SentimentScore(**row) for row in rows]


class PredictionRepository:
    """Repository for ML predictions."""

    @staticmethod
    def insert_prediction(
        ticker_id: int,
        target_date: date,
        direction: str,
        confidence: float,
        model_version: str = "rf_v1",
    ) -> int:
        """Insert or update a prediction."""
        with get_connection() as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO predictions (ticker_id, date, direction, confidence, model_version)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    direction = VALUES(direction),
                    confidence = VALUES(confidence),
                    model_version = VALUES(model_version)
            """
            cursor.execute(query, (ticker_id, target_date, direction, confidence, model_version))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            return affected

    @staticmethod
    def get_latest_prediction(ticker_id: int) -> Optional[Prediction]:
        """Get the most recent prediction for a ticker."""
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, ticker_id, date, direction, confidence, model_version, actual_direction
                FROM predictions 
                WHERE ticker_id = %s 
                ORDER BY date DESC LIMIT 1
                """,
                (ticker_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            return Prediction(**row) if row else None

    @staticmethod
    def update_actual_direction(ticker_id: int, target_date: date, actual: str) -> int:
        """Update the actual direction after market close."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE predictions SET actual_direction = %s WHERE ticker_id = %s AND date = %s",
                (actual, ticker_id, target_date),
            )
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            return affected


class ExchangeRateRepository:
    """Repository for exchange rates."""

    @staticmethod
    def insert_rate(
        base_currency: str, target_currency: str, rate: float, rate_date: date
    ) -> int:
        """Insert or update an exchange rate."""
        with get_connection() as conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO exchange_rates (base_currency, target_currency, rate, date)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE rate = VALUES(rate)
            """
            cursor.execute(query, (base_currency, target_currency, rate, rate_date))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            return affected

    @staticmethod
    def get_latest_rate(
        base_currency: str = "USD",
        target_currency: str = "JPY",
    ) -> Optional[float]:
        """Get the most recent exchange rate."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT rate FROM exchange_rates 
                WHERE base_currency = %s AND target_currency = %s 
                ORDER BY date DESC LIMIT 1
                """,
                (base_currency, target_currency),
            )
            result = cursor.fetchone()
            cursor.close()
            return float(result[0]) if result else None
