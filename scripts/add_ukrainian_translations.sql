-- Add Ukrainian translation column to news_articles
-- Run this on your server: mysql -u melvoice -p anime_stock < scripts/add_ukrainian_translations.sql

ALTER TABLE news_articles 
ADD COLUMN title_uk TEXT NULL AFTER title;

-- Add index for faster lookups
CREATE INDEX idx_news_title_uk ON news_articles(title_uk(100));

-- Add UAH exchange rate support (if not already in exchange_rates table)
-- The exchange_rates table already exists, no schema changes needed
