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
        "news_used_for_sentiment": "📰 News Used for Sentiment Analysis",
        "recent_headlines": "Recent headlines analyzed by AI to determine market sentiment:",
        "no_news_ticker": "No news articles found for this stock yet.",
        "news_used_for_sentiment": "📰 News Used for Sentiment Analysis",
        "recent_headlines": "Recent headlines analyzed by AI to determine market sentiment:",
        "no_news_ticker": "No news articles found for this stock yet.",
        
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
        "predictions_stats": "Predictions: {verified} verified, {future} future forecasts",
        "no_predictions_ticker": "No predictions found for {symbol}",
        "sentiment_not_calculated": "Sentiment analysis not yet calculated for these articles. Run the daily collection script to analyze.",
        
        # Footer
        "last_updated": "Last updated",
        "powered_by": "Powered by Animetrics AI 🏯",
        
        # Date formatting
        "date_format_short": "%b %d",  # Jan 31
        "date_format_full": "%Y-%m-%d %H:%M",
        
        # Info boxes
        "info_index_title": "What is Anime Index?",
        "info_index_text": "Composite metric tracking major anime companies. Each stock normalized to 100 at start, then averaged. See the industry trend at a glance! 🎯",
        "info_stocks_title": "AI-Powered Analysis",
        "info_stocks_text": "AI predicts next-day movements using company news & price patterns. Green ✅ = correct, Gray ❌ = wrong. Track accuracy!",
        "info_sentiment_title": "What is Sentiment?",
        "info_sentiment_text": "AI analyzes news headlines to gauge market mood. Score ranges from -1 (very negative 📉) to +1 (very positive 📈). Green bars = positive news, red = negative. Helps predict price movements!",
    },
    "uk": {
        # Page config
        "page_title": "Animetrics AI",
        
        # Control bar
        "currency": "Валюта",
        "view": "Що дивимось?",
        "view_index": "Індекс",
        "view_stocks": "Акції",
        "period": "Період",
        "stocks": "акцій",
        
        # Index page - headers
        "anime_industry_index": "📊 Аніме-індекс: Як справи в індустрії?",
        "latest_news": "📰 Свіженькі новини",
        "ai_predictions": "🤖 Що каже AI на завтра?",
        
        # Metrics
        "anime_index": "Аніме індекс",
        "market_sentiment": "Настрій ринку",
        "sentiment_bullish": "Бички рулять! 🐂",
        "sentiment_bearish": "Ведмеді атакують 🐻",
        "sentiment_neutral": "Та якось так... 😐",
        "sentiment_no_data": "Хмм, даних немає 🤷",
        "usd_jpy": "USD/JPY",
        "live": "Наживо",
        "tracked_stocks": "Відстежуємо",
        "active": "Активні",
        
        # Stocks page
        "select_stock": "Вибери свою акцію",
        "current_price": "Скільки коштує",
        "ai_forecast": "Що каже AI",
        "confidence": "впевненість",
        "no_prediction": "AI мовчить 🤐",
        "week_range": "Ціна за рік",
        "news_used_for_sentiment": "📰 Новини для аналізу настрою",
        "recent_headlines": "Останні заголовки, які AI проаналізував для визначення настрою:",
        "no_news_ticker": "Поки що новин для цієї акції немає.",
        "news_used_for_sentiment": "📰 Новини для аналізу настрою",
        "recent_headlines": "Останні заголовки, які AI проаналізував для визначення настрою:",
        "no_news_ticker": "Поки що новин для цієї акції немає.",
        
        # Chart labels
        "index_normalized": "Індекс аніме-індустрії (за базу взято 100)",
        "chart_sentiment": "Настрій ринку",
        "price": "Ціна",
        "volume": "Обсяг торгів",
        "sentiment": "Настрій",
        "index_value": "Значення індексу",
        "score": "Оцінка",
        
        # Table columns
        "symbol": "Тікер",
        "company": "Компанія",
        "sector": "Сектор",
        "direction": "Куди йде",
        "table_confidence": "Впевненість",
        
        # Messages
        "no_tickers": "Ой, тікерів не знайдено! 😱",
        "no_price_data": "Немає даних про ціни. Запусти колектор даних!",
        "no_news": "Поки що новин немає. Запусти парсер! 📡",
        "no_predictions": "AI ще не робив прогнози. Запусти предиктор! 🔮",
        "no_price_data_ticker": "Для {symbol} даних немає. Запусти колектор! 🚀",
        "predictions_stats": "Прогнози: {verified} перевірені, {future} майбутні",
        "no_predictions_ticker": "Немає прогнозів для {symbol}",
        "sentiment_not_calculated": "Аналіз настрою поки що не розраховано для цих статей. Запусти скрипт щоденного збору для аналізу.",
        
        # Footer
        "last_updated": "Оновлено",
        "powered_by": "Працює на Animetrics AI 🏯",
        
        # Date formatting
        "date_format_short": "%d.%m",  # 31.01
        "date_format_full": "%Y-%m-%d %H:%M",
        
        # Info boxes
        "info_index_title": "Що таке Аніме-індекс?",
        "info_index_text": "Збірний показник великих компаній аніме-індустрії. Всі акції нормалізовані до 100 на старті і усереднені. Бачиш тренд індустрії одним оком! 🎯",
        "info_stocks_title": "Аналіз на AI",
        "info_stocks_text": "AI передбачає рух на завтра за новинами та цінами. Зелена ✅ = вгадав, сіра ❌ = промах. Дивись точність!",
        "info_sentiment_title": "Що таке Настрій (Sentiment)?",
        "info_sentiment_text": "AI аналізує заголовки новин і визначає настрій ринку. Оцінка від -1 (дуже негатив 📉) до +1 (дуже позитив 📈). Зелені стовпчики = позитивні новини, червоні = негатив. Допомагає передбачити ціну!",
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
