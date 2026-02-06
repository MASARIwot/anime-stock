#!/bin/bash
# Anime Stock Data Refresh Script
# Run this daily to update all data on server: /opt/anime-stock
#
# Cron line for production:
# 0 18 * * * /opt/anime-stock/scripts/refresh_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/logs/refresh_$(date +%Y%m%d).log"

# Create logs directory if needed
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
source venv/bin/activate

echo "=== Anime Stock Refresh Started: $(date) ===" | tee -a "$LOG_FILE"

# Run daily collection
python -m anime_stock.scripts.daily_collect 2>&1 | tee -a "$LOG_FILE"

echo "=== Refresh Complete: $(date) ===" | tee -a "$LOG_FILE"
