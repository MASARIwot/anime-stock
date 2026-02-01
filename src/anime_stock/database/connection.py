"""Database connection management for Animetrics AI."""

import logging
from contextlib import contextmanager
from typing import Generator

import mysql.connector
from mysql.connector import Error
from mysql.connector.connection import MySQLConnection

from anime_stock.config import config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager with context support."""

    def __init__(self):
        self._connection: MySQLConnection | None = None

    def connect(self) -> MySQLConnection:
        """Establish database connection."""
        if self._connection is None or not self._connection.is_connected():
            try:
                self._connection = mysql.connector.connect(
                    host=config.database.host,
                    port=config.database.port,
                    user=config.database.username,
                    password=config.database.password,
                    database=config.database.database,
                    charset="utf8mb4",
                    collation="utf8mb4_unicode_ci",
                )
                logger.debug("Database connection established")
            except Error as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
        return self._connection

    def close(self):
        """Close database connection."""
        if self._connection and self._connection.is_connected():
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    def __enter__(self) -> MySQLConnection:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@contextmanager
def get_connection() -> Generator[MySQLConnection, None, None]:
    """Context manager for database connections."""
    db = DatabaseConnection()
    try:
        yield db.connect()
    finally:
        db.close()


def execute_query(query: str, params: tuple = None, fetch: bool = True) -> list[dict] | None:
    """Execute a query and optionally fetch results as dictionaries."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            conn.commit()
            return None
        finally:
            cursor.close()


def execute_many(query: str, params_list: list[tuple]) -> int:
    """Execute a query with multiple parameter sets. Returns affected rows."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
        finally:
            cursor.close()
