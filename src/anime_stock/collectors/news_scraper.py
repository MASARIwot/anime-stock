"""News scraper for anime industry RSS feeds."""

import logging
from datetime import datetime
from typing import Dict, Optional

import feedparser

from anime_stock.config import config
from anime_stock.database.repositories import NewsRepository

logger = logging.getLogger(__name__)

# Default RSS feeds for anime news
DEFAULT_FEEDS = {
    "ANN": "https://www.animenewsnetwork.com/all/rss.xml",
    "Crunchyroll": "https://www.crunchyroll.com/news/feed",
    "OtakuNews": "https://www.otakunews.com/rss/rss.xml",
}


class NewsScraper:
    """Scrapes anime industry news from RSS feeds."""

    def __init__(self, feeds: Optional[Dict[str, str]] = None):
        """
        Initialize the news scraper.
        
        Args:
            feeds: Dictionary mapping source name to RSS URL.
                   If None, uses default feeds.
        """
        self.feeds = feeds or DEFAULT_FEEDS
        self.news_repo = NewsRepository()

    def scrape_all(self, max_per_feed: int = 50) -> dict[str, int]:
        """
        Scrape news from all configured feeds.
        
        Args:
            max_per_feed: Maximum articles to fetch per feed.
        
        Returns:
            Dictionary mapping source name to number of new articles.
        """
        results = {}

        for source, url in self.feeds.items():
            try:
                count = self.scrape_feed(source, url, max_per_feed)
                results[source] = count
            except Exception as e:
                logger.error(f"Failed to scrape {source}: {e}")
                results[source] = 0

        return results

    def scrape_feed(self, source: str, url: str, max_articles: int = 50) -> int:
        """
        Scrape a single RSS feed.
        
        Args:
            source: Name of the news source.
            url: RSS feed URL.
            max_articles: Maximum articles to fetch.
        
        Returns:
            Number of new articles inserted.
        """
        logger.info(f"Scraping {source} from {url}...")

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error(f"Failed to parse feed {url}: {e}")
            return 0

        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Feed parsing issue for {source}: {feed.bozo_exception}")

        articles = []
        for entry in feed.entries[:max_articles]:
            try:
                article = self._parse_entry(source, entry)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse entry from {source}: {e}")
                continue

        if not articles:
            logger.warning(f"No valid articles found in {source}")
            return 0

        # Insert into database (duplicates are ignored by URL hash)
        count = NewsRepository.insert_articles(articles)
        logger.info(f"{source}: Found {len(articles)} articles, {count} new")
        return count

    def _parse_entry(self, source: str, entry) -> Optional[Dict]:
        """
        Parse a feedparser entry into an article dict.
        
        Args:
            source: News source name.
            entry: feedparser entry object.
        
        Returns:
            Article dictionary or None if invalid.
        """
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)

        if not title or not link:
            return None

        # Parse publication date
        published_at = None
        for date_field in ["published_parsed", "updated_parsed", "created_parsed"]:
            date_tuple = getattr(entry, date_field, None)
            if date_tuple:
                try:
                    published_at = datetime(*date_tuple[:6])
                    break
                except (TypeError, ValueError):
                    continue

        # Fallback to string parsing
        if not published_at:
            for date_field in ["published", "updated", "created"]:
                date_str = getattr(entry, date_field, None)
                if date_str:
                    published_at = self._parse_date_string(date_str)
                    if published_at:
                        break

        return {
            "source": source,
            "title": title[:500],  # Truncate to column limit
            "url": link[:1000],
            "published_at": published_at,
        }

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Try to parse a date string in various formats.
        
        Args:
            date_str: Date string to parse.
        
        Returns:
            datetime object or None if parsing failed.
        """
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None


def main():
    """CLI entry point for news scraping."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape anime news")
    parser.add_argument(
        "--max-per-feed",
        type=int,
        default=50,
        help="Maximum articles per feed (default: 50)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Scrape specific source only",
    )
    args = parser.parse_args()

    scraper = NewsScraper()

    if args.source:
        if args.source in scraper.feeds:
            count = scraper.scrape_feed(
                args.source, scraper.feeds[args.source], args.max_per_feed
            )
            print(f"Scraped {count} new articles from {args.source}")
        else:
            print(f"Unknown source: {args.source}")
            print(f"Available sources: {', '.join(scraper.feeds.keys())}")
    else:
        results = scraper.scrape_all(args.max_per_feed)
        print("\nScraping Results:")
        total = 0
        for source, count in results.items():
            print(f"  {source}: {count} new articles")
            total += count
        print(f"\nTotal: {total} new articles")


if __name__ == "__main__":
    main()
