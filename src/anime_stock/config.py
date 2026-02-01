"""Configuration loader for Animetrics AI."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_env_path)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str
    port: int
    database: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load database config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3308")),
            database=os.getenv("DB_DATABASE", "melvoice"),
            username=os.getenv("DB_USERNAME", "melvoice"),
            password=os.getenv("DB_PASSWORD", ""),
        )


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""

    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load OpenAI config from environment variables."""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )


@dataclass
class StreamlitConfig:
    """Streamlit server configuration."""

    host: str
    port: int

    @classmethod
    def from_env(cls) -> "StreamlitConfig":
        """Load Streamlit config from environment variables."""
        return cls(
            host=os.getenv("STREAMLIT_HOST", "0.0.0.0"),
            port=int(os.getenv("STREAMLIT_PORT", "8501")),
        )


@dataclass
class DataConfig:
    """Data collection configuration."""

    stock_tickers: list[str]
    news_feeds: list[str]
    exchange_rate_api_key: str

    @classmethod
    def from_env(cls) -> "DataConfig":
        """Load data config from environment variables."""
        tickers_str = os.getenv(
            "STOCK_TICKERS", "4816.T,9468.T,7832.T,9684.T,SONY,DIS,NFLX"
        )
        feeds_str = os.getenv(
            "NEWS_FEEDS",
            "https://www.animenewsnetwork.com/all/rss.xml,https://www.crunchyroll.com/news/feed",
        )
        return cls(
            stock_tickers=[t.strip() for t in tickers_str.split(",") if t.strip()],
            news_feeds=[f.strip() for f in feeds_str.split(",") if f.strip()],
            exchange_rate_api_key=os.getenv("EXCHANGE_RATE_API_KEY", ""),
        )


@dataclass
class Config:
    """Main configuration container."""

    database: DatabaseConfig
    openai: OpenAIConfig
    streamlit: StreamlitConfig
    data: DataConfig
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        """Load all configuration from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            streamlit=StreamlitConfig.from_env(),
            data=DataConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global configuration instance
config = Config.from_env()
