"""Backfill Ukrainian translations for existing news articles."""

import logging
from anime_stock.config import config
from anime_stock.collectors.news_scraper import NewsScraper
import mysql.connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Backfill translations for articles without title_uk."""
    
    # Connect to database
    conn = mysql.connector.connect(
        host=config.database.host,
        port=config.database.port,
        user=config.database.username,
        password=config.database.password,
        database=config.database.database,
    )
    
    scraper = NewsScraper()
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get articles without Ukrainian translations
        cursor.execute("""
            SELECT id, title 
            FROM news_articles 
            WHERE title_uk IS NULL 
            ORDER BY published_at DESC 
            LIMIT 30
        """)
        
        articles = cursor.fetchall()
        logger.info(f"Found {len(articles)} articles needing translation")
        
        translated_count = 0
        for article in articles:
            title_uk = scraper._translate_to_ukrainian(article['title'])
            if title_uk:
                cursor.execute(
                    "UPDATE news_articles SET title_uk = %s WHERE id = %s",
                    (title_uk, article['id'])
                )
                conn.commit()
                translated_count += 1
                logger.info(f"Translated [{translated_count}/{len(articles)}]: {article['title'][:50]}...")
        
        logger.info(f"Successfully translated {translated_count} articles")
        
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
