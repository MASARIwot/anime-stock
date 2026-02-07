"""Stock data collector using yfinance."""

import logging
from typing import Optional
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from anime_stock.database.repositories import (
    TickerRepository,
    PriceRepository,
    ExchangeRateRepository,
)

logger = logging.getLogger(__name__)


class StockCollector:
    """Collects stock price data from Yahoo Finance."""

    def __init__(self):
        self.ticker_repo = TickerRepository()
        self.price_repo = PriceRepository()
        self.exchange_repo = ExchangeRateRepository()

    def collect_all(self, backfill: bool = False) -> dict[str, int]:
        """
        Collect stock data for all active tickers.
        
        Args:
            backfill: If True, fetch 2 years of historical data.
                     If False, fetch only data since last recorded date.
        
        Returns:
            Dictionary mapping ticker symbol to number of records inserted.
        """
        tickers = TickerRepository.get_all_active()
        results = {}

        for ticker in tickers:
            try:
                count = self.collect_ticker(ticker.symbol, ticker.id, backfill)
                results[ticker.symbol] = count
            except Exception as e:
                logger.error(f"Failed to collect {ticker.symbol}: {e}")
                results[ticker.symbol] = 0

        return results

    def collect_ticker(self, symbol: str, ticker_id: int, backfill: bool = False) -> int:
        """
        Collect stock data for a single ticker.
        
        Args:
            symbol: Stock ticker symbol (e.g., "4816.T")
            ticker_id: Database ID for the ticker
            backfill: If True, fetch 2 years. Otherwise fetch from last date.
        
        Returns:
            Number of records inserted.
        """
        logger.info(f"Collecting data for {symbol}...")

        # Determine date range
        if backfill:
            start_date = None  # yfinance will use period="2y"
            period = "2y"
        else:
            last_date = PriceRepository.get_last_date(ticker_id)
            if last_date:
                # Fetch from day after last recorded date
                start_date = last_date + timedelta(days=1)
                if start_date >= date.today():
                    logger.info(f"{symbol}: Already up to date")
                    return 0
                period = None
            else:
                # No data exists, do initial backfill
                start_date = None
                period = "2y"

        # Fetch data from yfinance
        try:
            if period:
                data = yf.download(symbol, period=period, interval="1d", progress=False)
            else:
                data = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=date.today().strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                )
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return 0

        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return 0

        # Handle MultiIndex columns (yfinance returns this for single ticker too sometimes)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Prepare records for insertion
        records = []
        for idx, row in data.iterrows():
            record = {
                "ticker_id": ticker_id,
                "date": idx.date() if hasattr(idx, "date") else idx,
                "open": float(row["Open"]) if pd.notna(row.get("Open")) else None,
                "high": float(row["High"]) if pd.notna(row.get("High")) else None,
                "low": float(row["Low"]) if pd.notna(row.get("Low")) else None,
                "close": float(row["Close"]) if pd.notna(row.get("Close")) else 0,
                "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
            }
            records.append(record)

        # Insert into database
        count = PriceRepository.insert_prices(records)
        logger.info(f"{symbol}: Inserted {count} records")
        return count

    def fetch_exchange_rate(self) -> Optional[float]:
        """
        Fetch current USD/JPY and UAH exchange rates from free API.
        
        Returns:
            Exchange rate (USD to JPY) or None if failed.
        """
        import requests

        # Using Frankfurter API (free, no key required)
        url_jpy = "https://api.frankfurter.app/latest?from=USD&to=JPY"

        try:
            response = requests.get(url_jpy, timeout=10)
            response.raise_for_status()
            data = response.json()
            rate = data["rates"]["JPY"]

            # Store USD/JPY in database
            ExchangeRateRepository.insert_rate("USD", "JPY", rate, date.today())
            logger.info(f"USD/JPY rate: {rate}")
            
            # Also fetch UAH (Ukrainian Hryvnia) rate
            try:
                url_uah = "https://api.frankfurter.app/latest?from=UAH&to=USD"
                response_uah = requests.get(url_uah, timeout=10)
                if response_uah.ok:
                    data_uah = response_uah.json()
                    rate_usd_per_uah = data_uah["rates"]["USD"]
                    # Convert to UAH per USD
                    rate_uah = 1 / rate_usd_per_uah if rate_usd_per_uah > 0 else 40.0
                    ExchangeRateRepository.insert_rate("USD", "UAH", rate_uah, date.today())
                    logger.info(f"USD/UAH rate: {rate_uah:.2f}")
            except Exception as e:
                logger.warning(f"Failed to fetch UAH rate: {e}")
                # Default ~40 UAH per USD
                ExchangeRateRepository.insert_rate("USD", "UAH", 40.0, date.today())
            
            return rate
            logger.info(f"USD/JPY rate: {rate}")
            return rate

        except Exception as e:
            logger.error(f"Failed to fetch exchange rate: {e}")
            # Try to get cached rate from DB
            cached = ExchangeRateRepository.get_latest_rate("USD", "JPY")
            if cached:
                logger.info(f"Using cached rate: {cached}")
                return cached
            return None

    def get_current_rate(self) -> float:
        """Get current USD/JPY rate, fetching if needed."""
        rate = self.fetch_exchange_rate()
        return rate if rate else 150.0  # Fallback to reasonable default


def main():
    """CLI entry point for stock collection."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect anime stock data")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch 2 years of historical data",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Collect data for specific ticker only",
    )
    args = parser.parse_args()

    collector = StockCollector()

    if args.ticker:
        ticker = TickerRepository.get_by_symbol(args.ticker)
        if not ticker:
            print(f"Ticker {args.ticker} not found in database")
            sys.exit(1)
        count = collector.collect_ticker(ticker.symbol, ticker.id, args.backfill)
        print(f"Collected {count} records for {ticker.symbol}")
    else:
        results = collector.collect_all(args.backfill)
        print("\nCollection Results:")
        for symbol, count in results.items():
            print(f"  {symbol}: {count} records")

    # Always update exchange rate
    rate = collector.fetch_exchange_rate()
    if rate:
        print(f"\nExchange rate USD/JPY: {rate}")


if __name__ == "__main__":
    main()
