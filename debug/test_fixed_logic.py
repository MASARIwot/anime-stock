#!/usr/bin/env python3
from anime_stock.database.connection import get_connection

def test_fixed_logic():
    """Test the fixed prediction update logic."""
    
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
        
        if len(predictions) == 0:
            print("No predictions found - checking why...")
            
            # Check current date
            cursor.execute("SELECT CURDATE() as current_date")
            current_date = cursor.fetchone()['current_date']
            print(f"Current date (CURDATE()): {current_date}")
            
            # Check all predictions without actual_direction
            cursor.execute("""
                SELECT p.id, p.ticker_id, p.date as pred_date, p.direction, p.actual_direction
                FROM predictions p
                WHERE p.actual_direction IS NULL
                ORDER BY p.date DESC, p.ticker_id
                LIMIT 10
            """)
            all_null_predictions = cursor.fetchall()
            print(f"\nAll predictions without actual_direction:")
            for pred in all_null_predictions:
                print(f"  ID: {pred['id']}, Ticker: {pred['ticker_id']}, Date: {pred['pred_date']}, Direction: {pred['direction']}")
            return
        
        updated_count = 0
        
        for pred in predictions[:3]:  # Test first 3 predictions
            print(f"\nProcessing prediction ID {pred['id']}, ticker {pred['ticker_id']}, date {pred['pred_date']}")
            
            # Get the last available trading day before or on prediction date
            cursor.execute("""
                SELECT date, close
                FROM stock_prices
                WHERE ticker_id = %s
                  AND date <= %s
                ORDER BY date DESC
                LIMIT 1
            """, (pred['ticker_id'], pred['pred_date']))
            
            base_price_result = cursor.fetchone()
            print(f"Base price query result: {base_price_result}")
            
            if base_price_result:
                # Get the next available trading day after the base date
                cursor.execute("""
                    SELECT date, close
                    FROM stock_prices
                    WHERE ticker_id = %s
                      AND date > %s
                    ORDER BY date
                    LIMIT 1
                """, (pred['ticker_id'], base_price_result['date']))
                
                next_price_result = cursor.fetchone()
                print(f"Next price query result: {next_price_result}")
                
                if next_price_result:
                    pred_day_price = float(base_price_result['close'])
                    next_day_price = float(next_price_result['close'])
                    
                    # Determine actual direction
                    if next_day_price > pred_day_price:
                        actual_direction = 'UP'
                    elif next_day_price < pred_day_price:
                        actual_direction = 'DOWN'
                    else:
                        actual_direction = 'FLAT'
                    
                    print(f"  Base price: {pred_day_price} on {base_price_result['date']}")
                    print(f"  Next price: {next_day_price} on {next_price_result['date']}")
                    print(f"  Actual direction: {actual_direction}")
                    
                    print(f"  Would update prediction {pred['id']} with actual_direction = {actual_direction}")
                    updated_count += 1
                else:
                    print(f"  No next trading day data available")
            else:
                print(f"  No base price data available")
        
        cursor.close()
        
        print(f"\nWould have updated {updated_count} predictions")

if __name__ == "__main__":
    test_fixed_logic()