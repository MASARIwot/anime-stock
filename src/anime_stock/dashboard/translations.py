"""Translation system for Animetrics AI Dashboard."""

TRANSLATIONS = {
    "en": {
        # Page config
        "page_title": "Animetrics AI",
        
        # Control bar
        "currency": "Currency",
        "view": "View",
        "view_index": "Index",
        "view_stocks": "Stocks",
        "period": "Period",
        "stocks": "stocks",
        
        # Index page - headers
        "anime_industry_index": "📊 Anime Industry Composite Index",
        "latest_news": "📰 Latest News",
        "ai_predictions": "🤖 AI Predictions for Tomorrow",
        
        # Metrics
        "anime_index": "Anime Index",
        "market_sentiment": "Market Sentiment",
        "sentiment_bullish": "Bullish 🐂",
        "sentiment_bearish": "Bearish 🐻",
        "sentiment_neutral": "Neutral 😐",
        "sentiment_no_data": "No data",
        "usd_jpy": "USD/JPY",
        "live": "Live",
        "tracked_stocks": "Tracked Stocks",
        "active": "Active",
        
        # Stocks page
        "select_stock": "Select Stock",
        "current_price": "Current Price",
        "ai_forecast": "AI Forecast",
        "confidence": "confidence",
        "no_prediction": "No prediction",
        "week_range": "52W Range",
        
        # Chart labels
        "index_normalized": "Anime Industry Index (Normalized to 100)",
        "chart_sentiment": "Market Sentiment",
        "price": "Price",
        "volume": "Volume",
        "sentiment": "Sentiment",
        "index_value": "Index Value",
        "score": "Score",
        
        # Table columns
        "symbol": "Symbol",
        "company": "Company",
        "sector": "Sector",
        "direction": "Direction",
        "table_confidence": "Confidence",
        
        # Messages
        "no_tickers": "No tickers found.",
        "no_price_data": "No price data available. Run the collector first.",
        "no_news": "No news articles yet. Run the news scraper.",
        "no_predictions": "No predictions available. Run the predictor first.",
        "no_price_data_ticker": "No price data for {symbol}. Run the collector first.",
        
        # Footer
        "last_updated": "Last updated",
        "powered_by": "Powered by Animetrics AI 🏯",
        
        # Date formatting
        "date_format_short": "%b %d",  # Jan 31
        "date_format_full": "%Y-%m-%d %H:%M",
    },
    "uk": {
        # Page config
        "page_title": "Animetrics AI",
        
        # Control bar
        "currency": "Валюта",
        "view": "Вигляд",
        "view_index": "Індекс",
        "view_stocks": "Акції",
        "period": "Період",
        "stocks": "акцій",
        
        # Index page - headers
        "anime_industry_index": "📊 Композитний індекс аніме-індустрії",
        "latest_news": "📰 Останні новини",
        "ai_predictions": "🤖 Прогнози AI на завтра",
        
        # Metrics
        "anime_index": "Аніме індекс",
        "market_sentiment": "Настрій ринку",
        "sentiment_bullish": "Бичачий 🐂",
        "sentiment_bearish": "Ведмежий 🐻",
        "sentiment_neutral": "Нейтральний 😐",
        "sentiment_no_data": "Немає даних",
        "usd_jpy": "USD/JPY",
        "live": "Наживо",
        "tracked_stocks": "Відстежувані акції",
        "active": "Активні",
        
        # Stocks page
        "select_stock": "Оберіть акцію",
        "current_price": "Поточна ціна",
        "ai_forecast": "Прогноз AI",
        "confidence": "впевненість",
        "no_prediction": "Немає прогнозу",
        "week_range": "52-тижневий діапазон",
        
        # Chart labels
        "index_normalized": "Індекс аніме-індустрії (нормалізовано до 100)",
        "chart_sentiment": "Настрій ринку",
        "price": "Ціна",
        "volume": "Обсяг",
        "sentiment": "Настрій",
        "index_value": "Значення індексу",
        "score": "Оцінка",
        
        # Table columns
        "symbol": "Символ",
        "company": "Компанія",
        "sector": "Сектор",
        "direction": "Напрямок",
        "table_confidence": "Впевненість",
        
        # Messages
        "no_tickers": "Тікери не знайдено.",
        "no_price_data": "Немає даних про ціни. Запустіть збирач даних.",
        "no_news": "Немає новин. Запустіть парсер новин.",
        "no_predictions": "Немає прогнозів. Запустіть модуль прогнозування.",
        "no_price_data_ticker": "Немає даних про ціни для {symbol}. Запустіть збирач даних.",
        
        # Footer
        "last_updated": "Оновлено",
        "powered_by": "Працює на Animetrics AI 🏯",
        
        # Date formatting
        "date_format_short": "%d.%m",  # 31.01
        "date_format_full": "%Y-%m-%d %H:%M",
    }
}


def get_text(key: str, lang: str = "uk") -> str:
    """
    Get translated text for a given key and language.
    
    Args:
        key: Translation key
        lang: Language code ('en' or 'uk')
    
    Returns:
        Translated string, or the key itself if not found (fallback)
    """
    return TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS["en"].get(key, key))


def format_date(dt, lang: str = "uk", format_type: str = "short"):
    """
    Format date according to language preferences.
    
    Args:
        dt: datetime object
        lang: Language code
        format_type: 'short' or 'full'
    
    Returns:
        Formatted date string
    """
    if dt is None:
        return ""
    
    format_key = f"date_format_{format_type}"
    date_format = get_text(format_key, lang)
    
    return dt.strftime(date_format)
