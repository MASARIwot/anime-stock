"""ML price predictor using RandomForest."""

import logging
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from anime_stock.database.repositories import (
    TickerRepository,
    PriceRepository,
    SentimentRepository,
    PredictionRepository,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "rf_v1"


class PricePredictor:
    """Predicts stock price movement using RandomForest."""

    def __init__(self, n_estimators: int = 100, min_training_samples: int = 30):
        """
        Initialize the predictor.
        
        Args:
            n_estimators: Number of trees in the forest.
            min_training_samples: Minimum samples required for training.
        """
        self.n_estimators = n_estimators
        self.min_training_samples = min_training_samples
        self.model: Optional[RandomForestClassifier] = None
        self.trained_on: Optional[str] = None

    def prepare_features(self, ticker_id: int) -> Optional[pd.DataFrame]:
        """
        Prepare features for training/prediction.
        
        Combines stock prices with ticker-specific sentiment scores.
        
        Args:
            ticker_id: Database ID of the ticker.
        
        Returns:
            DataFrame with features or None if insufficient data.
        """
        # Get ticker info
        ticker = TickerRepository.get_by_id(ticker_id)
        if not ticker:
            logger.warning(f"Ticker {ticker_id} not found")
            return None

        # Get all prices for this ticker
        prices = PriceRepository.get_prices_for_ticker(ticker_id)
        if not prices:
            logger.warning(f"No price data for ticker {ticker_id}")
            return None

        # Convert to DataFrame
        price_df = pd.DataFrame([
            {
                "date": p.date,
                "close": float(p.close),
                "volume": p.volume or 0,
            }
            for p in prices
        ])
        price_df["date"] = pd.to_datetime(price_df["date"])
        price_df = price_df.set_index("date").sort_index()

        # Add technical indicators
        price_df["sma_5"] = price_df["close"].rolling(window=5).mean()
        price_df["sma_20"] = price_df["close"].rolling(window=20).mean()
        price_df["price_change"] = price_df["close"].pct_change()
        price_df["volatility"] = price_df["close"].rolling(window=10).std()

        # Get sentiment scores for this ticker
        sentiments = SentimentRepository.get_all_scores()
        if sentiments:
            # Filter for this ticker only
            ticker_sentiments = [s for s in sentiments if s.ticker == ticker.symbol]
            
            if ticker_sentiments:
                sent_df = pd.DataFrame([
                    {"date": s.date, "sentiment": float(s.score)}
                    for s in ticker_sentiments
                ])
                sent_df["date"] = pd.to_datetime(sent_df["date"])
                sent_df = sent_df.set_index("date")

                # Join with prices
                price_df = price_df.join(sent_df, how="left")
            else:
                # No sentiment for this ticker
                price_df["sentiment"] = 0.0
        else:
            price_df["sentiment"] = 0.0

        # Fill missing sentiment with 0 (neutral - no news)
        price_df["sentiment"] = price_df["sentiment"].fillna(0.0)

        # Create target: 1 if next day's close > today's close
        price_df["target"] = (price_df["close"].shift(-1) > price_df["close"]).astype(int)

        # Drop rows with NaN (from rolling windows and shift)
        price_df = price_df.dropna()

        if len(price_df) < self.min_training_samples:
            logger.warning(
                f"Insufficient data for ticker {ticker_id}: "
                f"{len(price_df)} < {self.min_training_samples}"
            )
            return None

        return price_df

    def train(self, ticker_id: int) -> Tuple[float, int]:
        """
        Train the model on historical data.
        
        Args:
            ticker_id: Database ID of the ticker to train on.
        
        Returns:
            Tuple of (accuracy, number of training samples).
        """
        df = self.prepare_features(ticker_id)
        if df is None:
            return 0.0, 0

        # Feature columns
        feature_cols = ["close", "sma_5", "sma_20", "price_change", "volatility", "sentiment"]
        X = df[feature_cols].values
        y = df["target"].values

        # Train/test split (use last 20% for testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        ticker = TickerRepository.get_by_id(ticker_id)
        self.trained_on = ticker.symbol if ticker else str(ticker_id)

        logger.info(
            f"Trained model for {self.trained_on}: "
            f"accuracy={accuracy:.2%}, samples={len(X_train)}"
        )

        return accuracy, len(X_train)

    def predict(self, ticker_id: int) -> Optional[Tuple[str, float]]:
        """
        Make prediction for tomorrow's price movement.
        
        Args:
            ticker_id: Database ID of the ticker.
        
        Returns:
            Tuple of (direction, confidence) or None if unable to predict.
        """
        if self.model is None:
            # Train first
            accuracy, samples = self.train(ticker_id)
            if self.model is None:
                return None

        df = self.prepare_features(ticker_id)
        if df is None:
            return None

        # Get latest features
        feature_cols = ["close", "sma_5", "sma_20", "price_change", "volatility", "sentiment"]
        latest = df[feature_cols].iloc[-1:].values

        # Predict
        prediction = self.model.predict(latest)[0]
        probabilities = self.model.predict_proba(latest)[0]
        confidence = probabilities[prediction]

        direction = "UP" if prediction == 1 else "DOWN"

        logger.info(
            f"Prediction for ticker {ticker_id}: {direction} ({confidence:.1%} confidence)"
        )

        return direction, confidence

    def predict_and_store(self, ticker_id: int) -> bool:
        """
        Make prediction and store in database.
        
        Args:
            ticker_id: Database ID of the ticker.
        
        Returns:
            True if prediction was made and stored.
        """
        result = self.predict(ticker_id)
        if result is None:
            return False

        direction, confidence = result

        # Store prediction for tomorrow
        prediction_date = date.today() + timedelta(days=1)
        PredictionRepository.insert_prediction(
            ticker_id=ticker_id,
            target_date=prediction_date,
            direction=direction,
            confidence=confidence,
            model_version=MODEL_VERSION,
        )

        logger.info(
            f"Stored prediction for {prediction_date}: {direction} ({confidence:.1%})"
        )
        return True

    def predict_all(self) -> dict[str, Tuple[str, float]]:
        """
        Make predictions for all active tickers.
        
        Returns:
            Dictionary mapping ticker symbol to (direction, confidence).
        """
        tickers = TickerRepository.get_all_active()
        results = {}

        for ticker in tickers:
            try:
                # Train fresh model for each ticker
                self.model = None
                result = self.predict(ticker.id)
                if result:
                    direction, confidence = result
                    results[ticker.symbol] = (direction, confidence)

                    # Store prediction
                    self.predict_and_store(ticker.id)

            except Exception as e:
                logger.error(f"Failed to predict for {ticker.symbol}: {e}")

        return results


def main():
    """CLI entry point for price prediction."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Predict anime stock prices")
    parser.add_argument(
        "--ticker",
        type=str,
        help="Predict for specific ticker only",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train model, don't store prediction",
    )
    args = parser.parse_args()

    predictor = PricePredictor()

    if args.ticker:
        ticker = TickerRepository.get_by_symbol(args.ticker)
        if not ticker:
            print(f"Ticker {args.ticker} not found")
            return

        if args.train_only:
            accuracy, samples = predictor.train(ticker.id)
            print(f"Trained model: accuracy={accuracy:.2%}, samples={samples}")
        else:
            result = predictor.predict(ticker.id)
            if result:
                direction, confidence = result
                print(f"Prediction for {args.ticker}: {direction} ({confidence:.1%})")
                predictor.predict_and_store(ticker.id)
            else:
                print("Unable to make prediction (insufficient data)")
    else:
        results = predictor.predict_all()
        print("\nPredictions for tomorrow:")
        for symbol, (direction, confidence) in results.items():
            emoji = "📈" if direction == "UP" else "📉"
            print(f"  {emoji} {symbol}: {direction} ({confidence:.1%})")


if __name__ == "__main__":
    main()
