#!/usr/bin/env python3
"""Update past predictions with actual results."""

import mysql.connector
from datetime import date, timedelta
from anime_stock.config import config

conn = mysql.connector.connect(
    host=config.database.host,
    port=config.database.port,
    user=config.database.username,
    password=config.database.password,
    database=config.database.database,
)

cursor = conn.cursor(dictionary=True)

# Get all predictions with NULL actual_direction where we have price data
cursor.execute("""
    SELECT p.id, p.ticker_id, p.date as pred_date, p.direction
    FROM predictions p
    WHERE p.actual_direction IS NULL
      AND p.date < CURDATE()
    ORDER BY p.date, p.ticker_id
""")

predictions = cursor.fetchall()
print(f"Found {len(predictions)} past predictions to update")

updated_count = 0
for pred in predictions:
    # Get the price on prediction date and next trading day
    cursor.execute("""
        SELECT date, close
        FROM stock_prices
        WHERE ticker_id = %s
          AND date >= %s
        ORDER BY date
        LIMIT 2
    """, (pred['ticker_id'], pred['pred_date']))
    
    prices = cursor.fetchall()
    
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
        
        # Update prediction
        cursor.execute("""
            UPDATE predictions
            SET actual_direction = %s
            WHERE id = %s
        """, (actual_direction, pred['id']))
        
        updated_count += 1
        print(f"  Updated prediction {pred['id']}: {pred['pred_date']} | Predicted: {pred['direction']} | Actual: {actual_direction}")

conn.commit()
print(f"\n✅ Updated {updated_count} predictions")

cursor.close()
conn.close()
