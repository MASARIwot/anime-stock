#!/bin/bash
# Setup cron job for Animetrics AI daily data collection
#
# This installs a cron job that runs daily at 6 PM (after markets close)
# to collect stock prices, news, sentiment, and generate predictions.
#
# Run this on your server: /opt/anime-stock/scripts/setup_cron.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

echo "🔧 Setting up Animetrics AI cron job..."
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Cron schedule: Daily at 18:00 (6 PM)"
echo ""

# Generate the cron line
CRON_LINE="0 18 * * * cd $PROJECT_DIR && source venv/bin/activate && python -m anime_stock.scripts.daily_collect >> $PROJECT_DIR/logs/cron.log 2>&1"

echo "This will add the following line to your crontab:"
echo ""
echo "$CRON_LINE"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "anime_stock.scripts.daily_collect"; then
    echo "⚠️  Cron job already exists!"
    echo ""
    echo "Current cron jobs related to Animetrics AI:"
    crontab -l | grep "anime_stock"
    echo ""
    read -p "Replace existing job? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    # Remove old job
    crontab -l | grep -v "anime_stock.scripts.daily_collect" | crontab -
fi

# Add new job
(crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -

echo "✅ Cron job installed successfully!"
echo ""
echo "The system will now run daily at 6 PM to:"
echo "  1. Collect stock prices"
echo "  2. Scrape news articles"
echo "  3. Analyze sentiment"
echo "  4. Update past predictions with actual results"
echo "  5. Generate new predictions"
echo ""
echo "Logs will be saved to: $PROJECT_DIR/logs/cron.log"
echo ""
echo "To view your cron jobs: crontab -l"
echo "To remove this job: crontab -e (then delete the line)"
echo ""
echo "📊 Dashboard should already be running at: http://localhost:8501"
