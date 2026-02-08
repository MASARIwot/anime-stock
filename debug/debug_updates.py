#!/usr/bin/env python3
from anime_stock.database.connection import get_connection
from datetime import datetime

def debug_prediction_update():
    """Debug the prediction update logic from daily_collect."""
    
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        
        # Get past predictions without actual results (same query as daily_collect)
        cursor.execute("""
            SELECT p.id, p.ticker_id, p.date as pred_date, p.direction
            FROM predictions p
            WHERE p.actual_direction IS NULL
              AND p.date < CURDATE()
            ORDER BY p.date, p.ticker_id
        """)
        
        predictions = cursor.fetchall()
        print(f"Found {len(predictions)} predictions to update")
        
        updated_count = 0
        
        for pred in predictions:
            print(f"\nProcessing prediction ID {pred['id']}, ticker {pred['ticker_id']}, date {pred['pred_date']}")
            
            # Get price on prediction date and next trading day (same logic as daily_collect)
            cursor.execute("""
                SELECT date, close
                FROM stock_prices
                WHERE ticker_id = %s
                  AND date >= %s
                ORDER BY date
                LIMIT 2
            """, (pred['ticker_id'], pred['pred_date']))
            
            prices = cursor.fetchall()
            print(f"Found {len(prices)} prices")
            for i, price in enumerate(prices):
                print(f"  Price {i+1}: {price['date']} = {price['close']}")
            
            if len(prices) >= 2:
                pred_day_price = float(prices[0]['close'])
                next_day_price = float(prices[1]['close'])
                
                # Determine actual direction
                if next_day_price > pred_day_price:
                    actual_direction = 'UP'
                elif next_day_price < pred_day_price:
                    actual_direction = 'DOWN'
                else:
                    actual_direction = 'FLAT'
                
                print(f"  Prediction day price: {pred_day_price}")
                print(f"  Next day price: {next_day_price}")
                print(f"  Actual direction: {actual_direction}")
                
                # Update prediction (DRY RUN - don't actually update)
                print(f"  Would update prediction {pred['id']} with actual_direction = {actual_direction}")
                updated_count += 1
            else:
                print(f"  Not enough price data to determine actual direction")
        
        cursor.close()
        
        print(f"\nWould have updated {updated_count} predictions")

if __name__ == "__main__":
    debug_prediction_update()