-- Migration: Add ticker column to news_articles and sentiment_scores
-- Purpose: Support stock-specific news from Yahoo Finance instead of generic anime industry news
-- Date: 2026-02-05

-- Add ticker column to news_articles
ALTER TABLE news_articles 
ADD COLUMN ticker VARCHAR(20);

-- Add ticker column to sentiment_scores  
ALTER TABLE sentiment_scores
ADD COLUMN ticker VARCHAR(20);

-- Create indexes for performance
CREATE INDEX idx_news_ticker_date ON news_articles(ticker, published_at);
CREATE INDEX idx_sentiment_ticker_date ON sentiment_scores(ticker, date);

-- Optional: Clear old anime industry news if you want fresh start
-- Uncomment the lines below to remove existing data:
-- DELETE FROM news_articles;
-- DELETE FROM sentiment_scores;
-- mysql -u melvoice -p melvoice < scripts/add_ticker_to_news.sql
