#!/usr/bin/env python3
from anime_stock.database.connection import get_connection
from datetime import datetime

with get_connection() as conn:
    cursor = conn.cursor(dictionary=True)
    
    # Check what predictions exist
    cursor.execute('SELECT COUNT(*) as total FROM predictions')
    total = cursor.fetchone()['total']
    print(f'Total predictions in database: {total}')
    
    # Check predictions without actual direction
    cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE actual_direction IS NULL')
    null_count = cursor.fetchone()['count']
    print(f'Predictions without actual_direction: {null_count}')
    
    # Check recent predictions
    cursor.execute('SELECT id, ticker_id, date, direction, actual_direction FROM predictions ORDER BY date DESC LIMIT 10')
    recent = cursor.fetchall()
    print(f'\nRecent predictions:')
    for pred in recent:
        print(f'  ID: {pred["id"]}, Ticker: {pred["ticker_id"]}, Date: {pred["date"]}, Direction: {pred["direction"]}, Actual: {pred["actual_direction"]}')
    
    # Check the specific query from daily_collect
    cursor.execute("""
        SELECT p.id, p.ticker_id, p.date as pred_date, p.direction
        FROM predictions p
        WHERE p.actual_direction IS NULL
          AND p.date < CURDATE()
        ORDER BY p.date, p.ticker_id
    """)
    past_predictions = cursor.fetchall()
    print(f'\nPast predictions to update: {len(past_predictions)}')
    for pred in past_predictions:
        print(f'  ID: {pred["id"]}, Ticker: {pred["ticker_id"]}, Date: {pred["pred_date"]}, Direction: {pred["direction"]}')
    
    cursor.close()