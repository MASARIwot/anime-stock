"""Daily data collection script for Animetrics AI.

This script is designed to be run via cron job daily.
It performs:
1. Stock price collection (incremental)
2. News scraping
3. Sentiment analysis
4. Price prediction

Usage:
    python -m anime_stock.scripts.daily_collect
    python -m anime_stock.scripts.daily_collect --backfill  # Initial 2-year data load
"""

import argparse
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_collect")


def run_collection(backfill: bool = False, skip_sentiment: bool = False):
    """
    Run the full data collection pipeline.
    
    Args:
        backfill: If True, fetch 2 years of historical stock data.
        skip_sentiment: If True, skip sentiment analysis (useful if no API key).
    """
    logger.info("=" * 60)
    logger.info("ANIMETRICS AI - DAILY DATA COLLECTION")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Step 1: Collect stock prices
    logger.info("\n📈 Step 1: Collecting stock prices...")
    try:
        from anime_stock.collectors.stock_collector import StockCollector
        
        collector = StockCollector()
        results = collector.collect_all(backfill=backfill)
        
        total_records = sum(results.values())
        logger.info(f"Stock collection complete: {total_records} records across {len(results)} tickers")
        
        # Also update exchange rate
        rate = collector.fetch_exchange_rate()
        if rate:
            logger.info(f"Exchange rate updated: 1 USD = {rate:.2f} JPY")
            
    except Exception as e:
        logger.error(f"Stock collection failed: {e}")

    # Step 2: Scrape news
    logger.info("\n📰 Step 2: Scraping news...")
    try:
        from anime_stock.collectors.news_scraper import NewsScraper
        
        scraper = NewsScraper()
        results = scraper.scrape_all()
        
        total_articles = sum(results.values())
        logger.info(f"News scraping complete: {total_articles} new articles")
        
    except Exception as e:
        logger.error(f"News scraping failed: {e}")

    # Step 3: Analyze sentiment
    if not skip_sentiment:
        logger.info("\n🧠 Step 3: Analyzing sentiment...")
        try:
            from anime_stock.analysis.sentiment import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            results = analyzer.process_unprocessed()
            
            logger.info(f"Sentiment analysis complete: {len(results)} dates processed")
            
        except ValueError as e:
            logger.warning(f"Sentiment analysis skipped: {e}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
    else:
        logger.info("\n🧠 Step 3: Skipping sentiment analysis (--skip-sentiment)")

    # Step 4: Generate predictions
    logger.info("\n🤖 Step 4: Generating predictions...")
    try:
        from anime_stock.analysis.predictor import PricePredictor
        
        predictor = PricePredictor()
        results = predictor.predict_all()
        
        logger.info(f"Prediction complete: {len(results)} tickers predicted")
        for symbol, (direction, confidence) in results.items():
            emoji = "📈" if direction == "UP" else "📉"
            logger.info(f"  {emoji} {symbol}: {direction} ({confidence:.1%})")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

    # Done
    logger.info("\n" + "=" * 60)
    logger.info(f"COLLECTION COMPLETE at {datetime.now().isoformat()}")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Animetrics AI - Daily Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regular daily run (incremental)
  python -m anime_stock.scripts.daily_collect

  # Initial setup with 2-year backfill
  python -m anime_stock.scripts.daily_collect --backfill

  # Skip sentiment if no OpenAI key
  python -m anime_stock.scripts.daily_collect --skip-sentiment
        """,
    )
    
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch 2 years of historical stock data (use for initial setup)",
    )
    
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment analysis (if no OpenAI API key)",
    )
    
    args = parser.parse_args()
    
    try:
        run_collection(
            backfill=args.backfill,
            skip_sentiment=args.skip_sentiment,
        )
    except KeyboardInterrupt:
        logger.info("\nCollection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Collection failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
