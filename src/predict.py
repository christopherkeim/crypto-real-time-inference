"""
Inference pipeline for machine learning and deep learning models.
"""

from typing import Dict, List, Callable
from pathlib import Path
import requests
from datetime import datetime, timedelta
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sktime.transformations.series.lag import Lag
from keras import Sequential
from keras.models import load_model


from src.feature_pipeline import (
    _get_time_signal_features,
    get_feature_engineering_pipeline,
)
from src.logger import get_console_logger
from src.paths import MODELS_DIR


logger = get_console_logger("model_inference")


def load_ml_model_from_name(model_name: str) -> Callable:
    """
    Takes an input name and returns a model.
    """
    MODEL_PATH: Path = MODELS_DIR / f"{model_name}_model.pkl"

    # Validate that the model exists locally
    if not MODEL_PATH.exists():
        raise NotImplementedError("Model not found 🔴")

    logger.info(f"Loading {model_name}_model.pkl ...")

    # Load the model into memory
    with open(MODELS_DIR / f"{model_name}_model.pkl", mode="rb") as f:
        ml_model = pickle.load(f)
    logger.info(f"{model_name} successfully loaded 🟢")

    return ml_model


def load_nn_model_from_name(model_name: str) -> Sequential:
    """
    Takes an input name and returns a model.
    """
    MODEL_PATH: Path = MODELS_DIR / f"{model_name}_model"

    # Validate that the model exists locally
    if not MODEL_PATH.exists():
        raise NotImplementedError("Model not found 🔴")

    logger.info(f"Loading {model_name}_model ...")

    # Load the model into memory
    nn_model = load_model(str(MODEL_PATH))
    logger.info(f"{model_name} successfully loaded 🟢")

    return nn_model


def download_data_for_t_hours(
    product_id: str, date_time_hour: str, t: int = 24
) -> pd.DataFrame:
    """
    Downloads raw OHLC candles for the specified cryptocurrency
    and date time hour + t hours into the past and returns them as a DataFrame.
    """
    target_time: str = f"{date_time_hour}:00:00"
    # Calculate the start time
    end = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%S")
    start = (end - timedelta(hours=t)).strftime("%Y-%m-%dT%H:%M:%S")

    # Call the Coinbase Exchange REST API for this product, datetime, and granularity
    URL: str = f"https://api.exchange.coinbase.com/products/{product_id}/"
    URL += f"candles?start={start}&end={target_time}&granularity=3600"
    r: requests.models.Response = requests.get(URL)
    data: List[List[int, float, float, float, float, float]] = r.json()

    # Transform list of lists to Pandas DataFrame and return it
    return pd.DataFrame(
        data, columns=["time", "low", "high", "open", "close", "volume"]
    )


def generate_scaled_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    SCALER_PATH: Path = MODELS_DIR / "X_scaler_model.pkl"

    # Extract column order to maintain it after scaling
    column_order: List[str] = list(X.columns)

    # Load the scaler from disk if it exists
    if SCALER_PATH.exists():
        logger.info(f"Loading X_scaler_model from: {SCALER_PATH} ...\n")
        with open(SCALER_PATH, mode="rb") as f:
            scaler: StandardScaler = pickle.load(f)

    # Fit a scaler to this data
    else:
        logger.info("Fitting scaler model to X ... 🪅\n")
        scaler: StandardScaler = StandardScaler()
        scaler.fit(X)
        logger.info("Scaler model successfully fit 🟢\n")

    # Transform the feature dataset
    logger.info("Transforming X features ... 🎭\n")
    X_scaled: np.ndarray = scaler.transform(X)
    logger.info("X successfully scaled 🟢\n")

    # Construct a DataFrame with the same columns as input
    X = pd.DataFrame(X_scaled, columns=column_order)

    return X


def get_feature_row_for_prediction(
    raw_ts_data: pd.DataFrame, window_seq_len: int = 24, step_size: int = 1
) -> pd.DataFrame:
    """
    This function ingests a raw candle, generates features for that
    candle, and returns a row of features for prediction as a DataFrame.
    """

    # Calculate hour (sin) and day (sin, cos) time features
    time_signal_data = _get_time_signal_features(raw_ts_data)
    # Convert seconds -> date time, sort by time, and drop duplicate time points
    time_signal_data["time"] = pd.to_datetime(time_signal_data["time"], unit="s")
    time_signal_data.sort_values(by=["time"], inplace=True)
    time_signal_data = time_signal_data.drop_duplicates("time", keep="first")
    time_signal_data.reset_index(drop=True, inplace=True)

    # Define a 1, 2, 3, ... -> 24 hour lag sequence for our x features
    x_time_steps: List[int] = [
        i for i in reversed(range(1, window_seq_len + 1, step_size))
    ]

    x_lag = Lag(x_time_steps)

    # Create lagged columns for closing price
    x_price = time_signal_data[["close"]].values
    x_lagged_price = x_lag.fit_transform(x_price)

    # Create lagged columns for volume
    x_volume = time_signal_data[["volume"]].values
    x_lagged_volume = x_lag.fit_transform(x_volume)

    # Price lagged features DataFrame
    price_features = pd.DataFrame(
        x_lagged_price,
        columns=[f"price_{i}_hour_ago" for i in x_time_steps],
    )

    # Volume lagged features DataFrame
    volume_features = pd.DataFrame(
        x_lagged_volume,
        columns=[f"volume_{i}_hour_ago" for i in x_time_steps],
    )

    # Full features DataFrame
    features_df = pd.concat([price_features, volume_features, time_signal_data], axis=1)

    # Drop NaN values -> single prediction row
    features_df = features_df.dropna(axis=0)

    # Set time as index
    features_df = features_df.set_index(["time"]).sort_index()

    # Feature engineering pipeline
    feature_engineering_pipeline = get_feature_engineering_pipeline()
    feature_engineering_pipeline.fit(features_df)
    X = feature_engineering_pipeline.transform(features_df)

    # Scale the full feature row
    X: pd.DataFrame = generate_scaled_features(X)

    return X


def predict(
    feature_row: pd.DataFrame,
    model_name: str = "cnn",
) -> float:
    """
    Takes a feature row and model name and returns that model's prediction
    for the cryptocurrency's price point the next hour.
    """
    MODEL_TYPE_MAPPINGS: Dict[str, str] = {"cnn": "nn", "lasso": "ml"}

    model_type: str = MODEL_TYPE_MAPPINGS[model_name]

    if model_type == "nn":
        # Load neural network model into memory
        model = load_nn_model_from_name(model_name)

        # Predict the next hour's price
        price_next_hour: float = model.predict(feature_row).flatten()[0]

    elif model_type == "ml":
        # Load machine learning model into memory
        model = load_ml_model_from_name(model_name)

        # Predict the next hour's price
        price_next_hour: float = model.predict(feature_row)[0]

    else:
        raise NotImplementedError("Model type is not implemented")

    return price_next_hour
