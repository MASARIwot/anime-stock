#!/usr/bin/env python3
"""Check what data we have in the database."""

import mysql.connector
from anime_stock.config import config

conn = mysql.connector.connect(
    host=config.database.host,
    port=config.database.port,
    user=config.database.username,
    password=config.database.password,
    database=config.database.database,
)
cursor = conn.cursor()

print('=== PREDICTIONS ===')
cursor.execute('SELECT COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest FROM predictions')
result = cursor.fetchone()
print(f"Count: {result[0]}, Earliest: {result[1]}, Latest: {result[2]}")

print('\n=== SENTIMENT_SCORES (with ticker) ===')
cursor.execute('SELECT COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest FROM sentiment_scores WHERE ticker IS NOT NULL')
result = cursor.fetchone()
print(f"Count: {result[0]}, Earliest: {result[1]}, Latest: {result[2]}")

print('\n=== NEWS_ARTICLES (with ticker) ===')
cursor.execute('SELECT COUNT(*) as count, MIN(published_at) as earliest, MAX(published_at) as latest FROM news_articles WHERE ticker IS NOT NULL')
result = cursor.fetchone()
print(f"Count: {result[0]}, Earliest: {result[1]}, Latest: {result[2]}")

print('\n=== SAMPLE PREDICTIONS (Latest 5) ===')
cursor.execute('''
    SELECT p.date, t.symbol, p.direction, p.actual_direction, p.confidence 
    FROM predictions p
    JOIN stock_tickers t ON p.ticker_id = t.id
    ORDER BY p.date DESC 
    LIMIT 5
''')
for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]:6s} | Predicted: {row[2]:4s} | Actual: {row[3] or 'NULL':4s} | Confidence: {row[4]:.0%}")

print('\n=== TICKERS WITH PREDICTIONS ===')
cursor.execute('''
    SELECT t.symbol, COUNT(*) as pred_count, 
           SUM(CASE WHEN actual_direction IS NOT NULL THEN 1 ELSE 0 END) as with_actual
    FROM predictions p
    JOIN stock_tickers t ON p.ticker_id = t.id
    GROUP BY t.symbol
''')
for row in cursor.fetchall():
    print(f"{row[0]:<6s} | Predictions: {row[1]:<3} | With actual result: {row[2]:<3}")

cursor.close()
conn.close()
