"""News scraper for Yahoo Finance stock-specific news."""

import logging
from datetime import datetime
from typing import Optional

import yfinance as yf
from openai import OpenAI

from anime_stock.config import config
from anime_stock.database.repositories import NewsRepository, TickerRepository

logger = logging.getLogger(__name__)


class NewsScraper:
    """Scrapes stock-specific news from Yahoo Finance."""

    def __init__(self):
        """Initialize the news scraper."""
        self.news_repo = NewsRepository()
        self.ticker_repo = TickerRepository()
        
        # Initialize OpenAI client for translations
        try:
            self.openai_client = OpenAI(api_key=config.openai.api_key)
            self.translation_enabled = True
        except Exception as e:
            logger.warning(f"OpenAI translation disabled: {e}")
            self.translation_enabled = False

    def scrape_all(self, max_per_ticker: int = 50) -> dict[str, int]:
        """
        Scrape news from Yahoo Finance for all active tickers.
        
        Args:
            max_per_ticker: Maximum articles to fetch per ticker.
        
        Returns:
            Dictionary mapping ticker symbol to number of new articles.
        """
        results = {}
        tickers = self.ticker_repo.get_all_active()

        for ticker in tickers:
            try:
                count = self.scrape_ticker(ticker.symbol, max_per_ticker)
                results[ticker.symbol] = count
            except Exception as e:
                logger.error(f"Failed to scrape news for {ticker.symbol}: {e}")
                results[ticker.symbol] = 0

        return results

    def scrape_ticker(self, symbol: str, max_articles: int = 50) -> int:
        """
        Scrape news for a specific ticker from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol.
            max_articles: Maximum articles to fetch.
        
        Returns:
            Number of new articles inserted.
        """
        logger.info(f"Scraping news for {symbol} from Yahoo Finance...")

        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            logger.debug(f"Yahoo Finance returned {len(news_items) if news_items else 0} raw items for {symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return 0

        if not news_items:
            logger.info(f"No news found for {symbol} (Yahoo Finance returned empty list)")
            return 0

        articles = []
        for item in news_items[:max_articles]:
            try:
                article = self._parse_yahoo_news(symbol, item)
                if article:
                    # Translate title to Ukrainian if enabled
                    if self.translation_enabled:
                        article['title_uk'] = self._translate_to_ukrainian(article['title'])
                    articles.append(article)
                else:
                    logger.debug(f"Skipped item for {symbol}: missing title or link")
            except Exception as e:
                logger.warning(f"Failed to parse news item for {symbol}: {e}")
                logger.debug(f"Item data: {item}")
                continue

        if not articles:
            logger.warning(f"No valid articles found for {symbol} (parsed 0/{len(news_items)} items)")
            return 0
            return 0

        # Insert into database (duplicates are ignored by URL hash)
        count = NewsRepository.insert_articles(articles)
        logger.info(f"{symbol}: Found {len(articles)} articles, {count} new")
        return count

    def _parse_yahoo_news(self, ticker: str, item: dict) -> Optional[dict]:
        """
        Parse a Yahoo Finance news item into an article dict.
        
        Args:
            ticker: Stock ticker symbol.
            item: Yahoo Finance news item dictionary.
        
        Returns:
            Article dictionary or None if invalid.
        """
        # Yahoo Finance API structure: item has 'content' object
        content = item.get("content", {})
        
        title = content.get("title")
        canonical_url = content.get("canonicalUrl", {})
        link = canonical_url.get("url") if isinstance(canonical_url, dict) else None
        
        # Fallback to other possible URL fields
        if not link:
            link = content.get("previewUrl") or item.get("link")
        
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"

        if not title or not link:
            return None

        # Parse publication date
        published_at = None
        pub_date = content.get("pubDate") or content.get("displayTime")
        if pub_date:
            try:
                from datetime import datetime
                # ISO 8601 format: "2026-02-06T22:01:00Z"
                published_at = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        return {
            "ticker": ticker,
            "source": publisher[:100],
            "title": title[:500],
            "url": link[:1000],
            "published_at": published_at,
        }

    def _translate_to_ukrainian(self, text: str) -> Optional[str]:
        """
        Translate English text to Ukrainian using OpenAI.
        
        Args:
            text: English text to translate.
        
        Returns:
            Ukrainian translation or None if translation fails.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the following news headline from English to Ukrainian. Provide ONLY the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=200,
                temperature=0.3,
            )
            
            translation = response.choices[0].message.content.strip()
            logger.debug(f"Translated: {text[:50]}... -> {translation[:50]}...")
            return translation[:500]  # Match title field length
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return None


def main():
    """CLI entry point for news scraping."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape stock news from Yahoo Finance")
    parser.add_argument(
        "--max-per-ticker",
        type=int,
        default=50,
        help="Maximum articles per ticker (default: 50)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Scrape specific ticker only",
    )
    args = parser.parse_args()

    scraper = NewsScraper()

    if args.ticker:
        count = scraper.scrape_ticker(args.ticker, args.max_per_ticker)
        print(f"Scraped {count} new articles for {args.ticker}")
    else:
        results = scraper.scrape_all(args.max_per_ticker)
        print("\nScraping Results:")
        total = 0
        for ticker, count in results.items():
            print(f"  {ticker}: {count} new articles")
            total += count
        print(f"\nTotal: {total} new articles")


if __name__ == "__main__":
    main()
