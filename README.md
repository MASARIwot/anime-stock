# 🏯 Animetrics AI

Anime stock market intelligence dashboard with LLM sentiment analysis and ML price predictions.

## Features

- 📈 **Stock Data Collection**: Tracks anime-related stocks (Toei, KADOKAWA, Bandai Namco, Sony, etc.)
- 📰 **News Scraping**: Aggregates headlines from Anime News Network, Crunchyroll, and more
- 🧠 **Sentiment Analysis**: GPT-4o-mini powered daily sentiment scoring
- 🤖 **ML Predictions**: RandomForest model correlates sentiment with price movement
- 💱 **Currency Toggle**: Live JPY↔USD conversion with real-time exchange rates
- 📊 **Anime Index**: Composite index tracking overall anime industry performance
- 🎨 **Clean Light Theme**: Streamlit dashboard with Plotly charts

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│  Streamlit Server   │     │   Refresh Script    │
│  (always running)   │     │   (cron or manual)  │
│                     │     │                     │
│  localhost:8501     │     │  runs periodically  │
└─────────┬───────────┘     └──────────┬──────────┘
          │                            │
          │         reads              │ writes
          ▼                            ▼
      ┌───────────────────────────────────┐
      │         MariaDB Database          │
      │  (stock_prices, news, sentiment)  │
      └───────────────────────────────────┘
```

The **dashboard server** and **data refresh** run in parallel:
- Dashboard serves the UI continuously and auto-refreshes its cache every 5 minutes
- Data refresh script runs periodically to update prices, news, and predictions

## Quick Start

**Python requirement:** 3.9+ (your 3.9.25 is supported)

### 1. Install Dependencies

```bash
cd anime-stock
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Initialize Database

```bash
# Run the SQL migration against your MariaDB
mysql -u melvoice -p melvoice < scripts/init_db.sql
```

### 4. Collect Initial Data (2-year backfill)

```bash
python -m anime_stock.scripts.daily_collect --backfill
```

### 5. Launch Dashboard

```bash
streamlit run src/anime_stock/dashboard/app.py --server.port 8501
```

## Project Structure

```
anime-stock/
├── scripts/
│   ├── init_db.sql          # Database migration
│   └── refresh_data.sh      # Data refresh script
├── src/anime_stock/
│   ├── config.py            # Configuration loader
│   ├── collectors/          # Data collection modules
│   │   ├── stock_collector.py
│   │   └── news_scraper.py
│   ├── analysis/            # AI/ML modules
│   │   ├── sentiment.py
│   │   └── predictor.py
│   ├── database/            # Database layer
│   │   ├── connection.py
│   │   └── repositories.py
│   ├── dashboard/           # Streamlit app
│   │   └── app.py
│   └── scripts/             # CLI scripts
│       └── daily_collect.py
└── logs/                    # Refresh logs (gitignored)
```

## Running

### Dashboard Server (runs continuously)

```bash
source venv/bin/activate
streamlit run src/anime_stock/dashboard/app.py --server.port 8501
```

### Data Refresh (run manually or via cron)

```bash
# Quick refresh script
./scripts/refresh_data.sh

# Or directly
python -m anime_stock.scripts.daily_collect
```

## Deployment

### Cron Job (Daily Data Refresh)

```bash
crontab -e

# Add: Run daily at 9 AM (after Tokyo market opens)
0 9 * * * /path/to/anime-stock/scripts/refresh_data.sh
```

### Systemd Service (Dashboard)

```bash
sudo cp anime_stock.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable anime_stock
sudo systemctl start anime_stock
```

## PHP Integration

Add to your Laravel routes:

```php
Route::get('/stock', 'StockController@index');
```

Create the view with iframe:

```blade
<iframe src="http://localhost:8501" width="100%" height="800px" frameborder="0"></iframe>
```

## License

MIT
