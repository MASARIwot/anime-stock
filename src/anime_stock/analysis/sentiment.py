"""Sentiment analysis using OpenAI GPT-4o-mini."""

import logging
from datetime import date
from typing import Optional

from openai import OpenAI

from anime_stock.config import config
from anime_stock.database.repositories import NewsRepository, SentimentRepository

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial analyst specializing in stock market sentiment analysis. 
Your task is to analyze news headlines for a specific company and rate the sentiment.

Consider:
- Positive earnings, revenue growth = positive
- New product launches, partnerships = positive  
- Strong financial results, analyst upgrades = positive
- Losses, declining revenue = negative
- Lawsuits, scandals, management issues = negative
- Regulatory problems, recalls = negative
- Neutral industry updates = 0

Return ONLY a single number between -1.0 (very negative) and 1.0 (very positive).
No text, no explanation, just the number with up to 2 decimal places."""


class SentimentAnalyzer:
    """Analyzes news sentiment using OpenAI GPT-4o-mini."""

    def __init__(self):
        """Initialize the sentiment analyzer."""
        if not config.openai.api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.client = OpenAI(api_key=config.openai.api_key)
        self.model = config.openai.model

    def analyze_headlines(self, headlines: list[str]) -> float:
        """
        Analyze a list of headlines and return sentiment score.
        
        Args:
            headlines: List of news headlines to analyze.
        
        Returns:
            Sentiment score from -1.0 to 1.0.
        """
        if not headlines:
            return 0.0

        # Combine headlines with separator
        combined_text = " | ".join(headlines)

        # Truncate if too long (API limit considerations)
        if len(combined_text) > 10000:
            combined_text = combined_text[:10000] + "..."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": combined_text},
                ],
                temperature=0,
                max_tokens=10,
            )

            score_str = response.choices[0].message.content.strip()
            score = float(score_str)

            # Clamp to valid range
            score = max(-1.0, min(1.0, score))
            
            logger.debug(f"Sentiment score for {len(headlines)} headlines: {score}")
            return score

        except ValueError as e:
            logger.error(f"Failed to parse sentiment score: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return 0.0

    def process_date(self, target_date: date, ticker: str) -> Optional[float]:
        """
        Process news articles for a specific date and ticker.
        
        Args:
            target_date: Date to process.
            ticker: Stock ticker symbol.
        
        Returns:
            Sentiment score or None if no articles found.
        """
        # Get articles for this date and ticker
        articles = NewsRepository.get_articles_for_date(target_date, ticker=ticker)

        if not articles:
            logger.info(f"No articles found for {ticker} on {target_date}")
            return None

        headlines = [a.title for a in articles]
        logger.info(f"Analyzing {len(headlines)} headlines for {ticker} on {target_date}")

        # Analyze sentiment
        score = self.analyze_headlines(headlines)

        # Store in database
        raw_headlines = " | ".join(headlines[:20])  # Store first 20 for debugging
        SentimentRepository.insert_score(
            target_date=target_date,
            score=score,
            model_used=self.model,
            headlines_count=len(headlines),
            ticker=ticker,
            raw_headlines=raw_headlines,
        )

        logger.info(f"Stored sentiment for {ticker} on {target_date}: {score:.2f}")
        return score

    def process_unprocessed(self) -> dict[tuple[date, str], float]:
        """
        Process all (date, ticker) pairs that have news but no sentiment score.
        
        Returns:
            Dictionary mapping (date, ticker) to sentiment score.
        """
        unprocessed = NewsRepository.get_unprocessed_dates()
        
        if not unprocessed:
            logger.info("No unprocessed dates found")
            return {}

        logger.info(f"Processing {len(unprocessed)} unprocessed (date, ticker) pairs")
        
        results = {}
        for target_date, ticker in unprocessed:
            score = self.process_date(target_date, ticker)
            if score is not None:
                results[(target_date, ticker)] = score

        return results


def main():
    """CLI entry point for sentiment analysis."""
    import argparse
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Analyze stock news sentiment")
    parser.add_argument(
        "--date",
        type=str,
        help="Analyze specific date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Ticker symbol (required with --date)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all unprocessed dates",
    )
    args = parser.parse_args()

    try:
        analyzer = SentimentAnalyzer()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    if args.date:
        if not args.ticker:
            print("Error: --ticker is required when using --date")
            return
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        score = analyzer.process_date(target_date, args.ticker)
        if score is not None:
            print(f"Sentiment for {args.ticker} on {target_date}: {score:.2f}")
        else:
            print(f"No articles found for {args.ticker} on {target_date}")
    elif args.all:
        results = analyzer.process_unprocessed()
        print(f"\nProcessed {len(results)} (date, ticker) pairs:")
        for (d, t), score in sorted(results.items()):
            print(f"  {d} {t}: {score:.2f}")
    else:
        print("Error: Use --date with --ticker, or --all")
        print("Example: python -m anime_stock.analysis.sentiment --date 2026-02-05 --ticker AAPL")


if __name__ == "__main__":
    main()
