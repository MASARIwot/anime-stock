-- ============================================================================
-- Animetrics AI - Database Migration
-- Run this SQL against the 'melvoice' database
-- ============================================================================

-- Table: stock_tickers
-- Stores the list of anime-related stock tickers we track
CREATE TABLE IF NOT EXISTS stock_tickers (
    id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE COMMENT 'Ticker symbol (e.g., 4816.T, SONY)',
    company_name VARCHAR(255) NOT NULL COMMENT 'Company name',
    exchange VARCHAR(50) NOT NULL COMMENT 'Exchange (TSE, NYSE, NASDAQ)',
    sector VARCHAR(100) DEFAULT 'anime' COMMENT 'Sector classification',
    currency VARCHAR(3) NOT NULL DEFAULT 'JPY' COMMENT 'Native currency (JPY, USD)',
    active TINYINT(1) NOT NULL DEFAULT 1 COMMENT 'Whether to track this ticker',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_active (active),
    INDEX idx_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: stock_prices
-- Daily OHLCV data for each ticker
CREATE TABLE IF NOT EXISTS stock_prices (
    id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT(10) UNSIGNED NOT NULL,
    date DATE NOT NULL COMMENT 'Trading date',
    open DECIMAL(15, 4) COMMENT 'Opening price',
    high DECIMAL(15, 4) COMMENT 'High price',
    low DECIMAL(15, 4) COMMENT 'Low price',
    close DECIMAL(15, 4) NOT NULL COMMENT 'Closing price',
    volume BIGINT UNSIGNED COMMENT 'Trading volume',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_ticker_date (ticker_id, date),
    INDEX idx_date (date),
    INDEX idx_ticker_id (ticker_id),
    CONSTRAINT fk_stock_prices_ticker 
        FOREIGN KEY (ticker_id) REFERENCES stock_tickers(id) 
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: news_articles
-- Scraped anime industry news headlines
CREATE TABLE IF NOT EXISTS news_articles (
    id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(100) NOT NULL COMMENT 'News source (ANN, Crunchyroll, etc.)',
    title VARCHAR(500) NOT NULL COMMENT 'Article headline',
    url VARCHAR(1000) NOT NULL COMMENT 'Article URL',
    url_hash VARCHAR(64) AS (SHA2(url, 256)) STORED UNIQUE COMMENT 'Hash for uniqueness check',
    published_at DATETIME COMMENT 'Article publication date',
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'When we scraped it',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_published_at (published_at),
    INDEX idx_source (source),
    INDEX idx_scraped_at (scraped_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: sentiment_scores
-- Daily aggregated sentiment scores from LLM analysis
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL UNIQUE COMMENT 'Date of sentiment score',
    score DECIMAL(5, 4) NOT NULL COMMENT 'Sentiment score (-1.0 to 1.0)',
    model_used VARCHAR(50) DEFAULT 'gpt-4o-mini' COMMENT 'LLM model used',
    headlines_count INT UNSIGNED DEFAULT 0 COMMENT 'Number of headlines analyzed',
    raw_headlines TEXT COMMENT 'Headlines that were analyzed (for debugging)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_date (date),
    INDEX idx_score (score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: predictions
-- ML model predictions for price movement
CREATE TABLE IF NOT EXISTS predictions (
    id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT(10) UNSIGNED NOT NULL,
    date DATE NOT NULL COMMENT 'Prediction date',
    direction ENUM('UP', 'DOWN') NOT NULL COMMENT 'Predicted direction',
    confidence DECIMAL(5, 4) NOT NULL COMMENT 'Prediction confidence (0-1)',
    model_version VARCHAR(50) DEFAULT 'rf_v1' COMMENT 'Model version identifier',
    actual_direction ENUM('UP', 'DOWN') COMMENT 'Actual direction (filled next day)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_ticker_date (ticker_id, date),
    INDEX idx_date (date),
    INDEX idx_ticker_id (ticker_id),
    CONSTRAINT fk_predictions_ticker 
        FOREIGN KEY (ticker_id) REFERENCES stock_tickers(id) 
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: exchange_rates
-- Cache for currency exchange rates
CREATE TABLE IF NOT EXISTS exchange_rates (
    id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    base_currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    target_currency VARCHAR(3) NOT NULL DEFAULT 'JPY',
    rate DECIMAL(15, 6) NOT NULL COMMENT 'Exchange rate',
    date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_currencies_date (base_currency, target_currency, date),
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Initial Data: Default tickers to track
-- ============================================================================

INSERT INTO stock_tickers (symbol, company_name, exchange, sector, currency) VALUES
    ('4816.T', 'Toei Animation', 'TSE', 'anime', 'JPY'),
    ('9468.T', 'KADOKAWA', 'TSE', 'anime', 'JPY'),
    ('7832.T', 'Bandai Namco Holdings', 'TSE', 'gaming', 'JPY'),
    ('9684.T', 'Square Enix Holdings', 'TSE', 'gaming', 'JPY'),
    ('SONY', 'Sony Group Corporation', 'NYSE', 'entertainment', 'USD'),
    ('DIS', 'The Walt Disney Company', 'NYSE', 'entertainment', 'USD'),
    ('NFLX', 'Netflix Inc', 'NASDAQ', 'streaming', 'USD'),
    ('9602.T', 'Toho Co Ltd', 'TSE', 'anime', 'JPY'),
    ('3659.T', 'Nexon Co Ltd', 'TSE', 'gaming', 'JPY'),
    ('7974.T', 'Nintendo Co Ltd', 'TSE', 'gaming', 'JPY')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- Verification: Show created tables
-- ============================================================================

SELECT 'Tables created successfully!' AS status;
SHOW TABLES LIKE 'stock_%';
SHOW TABLES LIKE 'news_%';
SHOW TABLES LIKE 'sentiment_%';
SHOW TABLES LIKE 'predictions';
SHOW TABLES LIKE 'exchange_%';
