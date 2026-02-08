#!/usr/bin/env python3
from anime_stock.database.connection import get_connection

def check_specific_query():
    """Test the exact query from daily_collect for ticker 1 and date 2026-02-07."""
    
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        
        # Test the exact query for one prediction
        ticker_id = 1
        pred_date = '2026-02-07'
        
        print(f"Testing query for ticker {ticker_id}, prediction date {pred_date}")
        
        cursor.execute("""
            SELECT date, close
            FROM stock_prices
            WHERE ticker_id = %s
              AND date >= %s
            ORDER BY date
            LIMIT 2
        """, (ticker_id, pred_date))
        
        prices = cursor.fetchall()
        print(f"Query result: {len(prices)} prices")
        for price in prices:
            print(f"  {price['date']}: {price['close']}")
        
        # Let's also check what dates we have around this prediction date
        cursor.execute("""
            SELECT date, close
            FROM stock_prices
            WHERE ticker_id = %s
              AND date BETWEEN DATE_SUB(%s, INTERVAL 5 DAY) AND DATE_ADD(%s, INTERVAL 5 DAY)
            ORDER BY date
        """, (ticker_id, pred_date, pred_date))
        
        nearby_prices = cursor.fetchall()
        print(f"\nNearby prices (+/- 5 days):")
        for price in nearby_prices:
            print(f"  {price['date']}: {price['close']}")
        
        cursor.close()

if __name__ == "__main__":
    check_specific_query()