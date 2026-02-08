#!/usr/bin/env python3
from anime_stock.database.connection import get_connection

def check_stock_prices():
    """Check what stock price data exists."""
    
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        
        # Check the date range of stock prices
        cursor.execute("SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_prices")
        date_range = cursor.fetchone()
        print(f"Stock price date range: {date_range['min_date']} to {date_range['max_date']}")
        
        # Check recent stock prices
        cursor.execute("SELECT ticker_id, date, close FROM stock_prices ORDER BY date DESC LIMIT 20")
        recent_prices = cursor.fetchall()
        print(f"\nRecent stock prices:")
        for price in recent_prices:
            print(f"  Ticker {price['ticker_id']}: {price['date']} = {price['close']}")
        
        # Check if we have prices for prediction dates
        cursor.execute("""
            SELECT DISTINCT p.date, COUNT(*) as price_count
            FROM predictions p
            LEFT JOIN stock_prices sp ON sp.ticker_id = p.ticker_id AND sp.date = p.date
            WHERE p.actual_direction IS NULL AND p.date < CURDATE()
            GROUP BY p.date
            ORDER BY p.date
        """)
        prediction_dates = cursor.fetchall()
        print(f"\nPrice data for prediction dates:")
        for row in prediction_dates:
            print(f"  {row['date']}: {row['price_count']} price records")
        
        cursor.close()

if __name__ == "__main__":
    check_stock_prices()