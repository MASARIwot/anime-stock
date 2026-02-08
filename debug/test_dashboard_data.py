"""Test dashboard data loading."""

from anime_stock.database.repositories import NewsRepository, TickerRepository, ExchangeRateRepository

# Test 1: News with translations
print("=" * 60)
print("TEST 1: Latest News Articles")
print("=" * 60)
articles = NewsRepository.get_latest_articles(10)
tickers = TickerRepository.get_all_active()
tracked_symbols = {t.symbol for t in tickers}

filtered = [a for a in articles if a.ticker and a.ticker in tracked_symbols][:10]

print(f"Total articles fetched: {len(articles)}")
print(f"Filtered for tracked tickers: {len(filtered)}")
print(f"\nFirst 3 articles:")
for i, a in enumerate(filtered[:3], 1):
    print(f"\n{i}. [{a.ticker}] {a.title[:50]}...")
    print(f"   UK: {a.title_uk[:50] if a.title_uk else 'NO TRANSLATION'}...")

# Test 2: Exchange rates
print("\n" + "=" * 60)
print("TEST 2: Exchange Rates")
print("=" * 60)
rate_usd_jpy = ExchangeRateRepository.get_latest_rate("USD", "JPY")
rate_usd_uah = ExchangeRateRepository.get_latest_rate("USD", "UAH")
print(f"USD -> JPY: {rate_usd_jpy if rate_usd_jpy else 'NOT FOUND (will use fallback 150)'}")
print(f"USD -> UAH: {rate_usd_uah if rate_usd_uah else 'NOT FOUND (will use fallback 40)'}")

# Test 3: Currency symbols
print("\n" + "=" * 60)
print("TEST 3: Currency Symbol Logic")
print("=" * 60)
for currency in ["USD", "JPY", "UAH"]:
    if currency == "JPY":
        symbol = "¥"
    elif currency == "UAH":
        symbol = "₴"
    else:
        symbol = "$"
    print(f"{currency}: {symbol}")
