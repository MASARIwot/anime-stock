"""Sentiment analysis using OpenAI GPT-4o-mini."""

import logging
from datetime import date

from openai import OpenAI

from anime_stock.config import config
from anime_stock.database.repositories import NewsRepository, SentimentRepository

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial analyst specializing in Anime & Manga markets. 
Your task is to analyze news headlines and rate the overall market sentiment.

Consider:
- New anime announcements = positive
- Streaming deals, partnerships = positive  
- Box office success, award wins = positive
- Delays, cancellations = negative
- Studio troubles, layoffs = negative
- Controversy, legal issues = negative
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

    def process_date(self, target_date: date) -> float | None:
        """
        Process all news articles for a specific date.
        
        Args:
            target_date: Date to process.
        
        Returns:
            Sentiment score or None if no articles found.
        """
        # Get articles for this date
        articles = NewsRepository.get_articles_for_date(target_date)

        if not articles:
            logger.info(f"No articles found for {target_date}")
            return None

        headlines = [a.title for a in articles]
        logger.info(f"Analyzing {len(headlines)} headlines for {target_date}")

        # Analyze sentiment
        score = self.analyze_headlines(headlines)

        # Store in database
        raw_headlines = " | ".join(headlines[:20])  # Store first 20 for debugging
        SentimentRepository.insert_score(
            target_date=target_date,
            score=score,
            model_used=self.model,
            headlines_count=len(headlines),
            raw_headlines=raw_headlines,
        )

        logger.info(f"Stored sentiment for {target_date}: {score:.2f}")
        return score

    def process_unprocessed(self) -> dict[date, float]:
        """
        Process all dates that have news but no sentiment score.
        
        Returns:
            Dictionary mapping date to sentiment score.
        """
        unprocessed_dates = NewsRepository.get_unprocessed_dates()
        
        if not unprocessed_dates:
            logger.info("No unprocessed dates found")
            return {}

        logger.info(f"Processing {len(unprocessed_dates)} unprocessed dates")
        
        results = {}
        for target_date in unprocessed_dates:
            score = self.process_date(target_date)
            if score is not None:
                results[target_date] = score

        return results


def main():
    """CLI entry point for sentiment analysis."""
    import argparse
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Analyze anime news sentiment")
    parser.add_argument(
        "--date",
        type=str,
        help="Analyze specific date (YYYY-MM-DD)",
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
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        score = analyzer.process_date(target_date)
        if score is not None:
            print(f"Sentiment for {target_date}: {score:.2f}")
        else:
            print(f"No articles found for {target_date}")
    elif args.all:
        results = analyzer.process_unprocessed()
        print(f"\nProcessed {len(results)} dates:")
        for d, score in sorted(results.items()):
            print(f"  {d}: {score:.2f}")
    else:
        # Default: process today
        today = date.today()
        score = analyzer.process_date(today)
        if score is not None:
            print(f"Today's sentiment: {score:.2f}")
        else:
            print("No articles found for today")


if __name__ == "__main__":
    main()
