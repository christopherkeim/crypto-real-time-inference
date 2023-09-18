"""
Prepocessing / feature engineering steps after data has been scraped.

1. Transform raw time series data into (features, target) format by generating
   lag features over price and volume

2. Calculate window features (moving average, moving std) over price and volume

3. Calculate technical inidicator features

Preprocessed datasets can be versioned with Weights & Biases and saved locally 
if specified.

This code can be used as a module in another script or as a CLI tool. 
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import os
import click
import wandb

import numpy as np
import pandas as pd
from sktime.transformations.series.lag import Lag
import ta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.paths import DATA_DIR
from src.logger import get_console_logger

logger = get_console_logger("dataset_preprocessing")


def transform_ts_data_into_lagged_features_and_target(
    path_to_input: Optional[Path] = DATA_DIR / "BTC-USD_ohlc_data.parquet",
    window_seq_len: Optional[int] = 24,
    step_size: Optional[int] = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculates lag features and transforms raw data from time series format
    into a (features, target) format that can be used to train supervised ML models.
    """
    # Load the parquet file
    ts_data = pd.read_parquet(path_to_input)
    # Calculate hour (sin) and day (sin, cos) time features
    ts_data = _get_time_signal_features(ts_data)
    # Convert seconds -> date time, sort by time, and drop duplicate time points
    ts_data["time"] = pd.to_datetime(ts_data["time"], unit="s")
    ts_data.sort_values(by=["time"], inplace=True)
    ts_data = ts_data.drop_duplicates("time", keep="first")
    ts_data.reset_index(drop=True, inplace=True)

    # Define a 1, 2, 3, ... -> 24 hour lag sequence for our x features
    x_time_steps: List[int] = [
        i for i in reversed(range(1, window_seq_len + 1, step_size))
    ]

    x_lag = Lag(x_time_steps)

    # Create lagged columns for closing price
    x_price = ts_data[["close"]].values
    x_lagged_price = x_lag.fit_transform(x_price)

    # Create lagged columns for volume
    x_volume = ts_data[["volume"]].values
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
    features_df = pd.concat([price_features, volume_features, ts_data], axis=1)

    # Define a -1 hour (1 hour in future) lag for our target variable y
    y_lag = Lag(-1)

    y_price = features_df[["close"]].values
    y_lagged_price = y_lag.fit_transform(y_price)[1:]

    # Targets -1 (1 hour in future) lagged DataFrame
    targets_df = pd.DataFrame(y_lagged_price, columns=["target_price_next_hour"])

    # Concatenate DataFrames to drop NaN values
    full_df = pd.concat([features_df, targets_df], axis=1)
    full_df = full_df.dropna(axis=0)

    # Set time as index
    full_df = full_df.set_index(["time"]).sort_index()

    # Now extract our features and targets
    features = full_df.drop("target_price_next_hour", axis=1)
    targets = full_df["target_price_next_hour"]

    return features, targets


def _get_time_signal_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 3 new time signal features ("sin_hour", "sin_day"
    "cos_day") and adds these columns to our DataFrame.
    """
    NUMBER_OF_SECONDS_IN_HOUR = 60 * 60
    NUMBER_OF_SECONDS_IN_DAY = 60 * 60 * 24
    # Hour sin signal
    X["sin_hour"] = np.sin(X["time"] * (2 * np.pi / NUMBER_OF_SECONDS_IN_HOUR))
    # Day sin and cosine signals
    X["sin_day"] = np.sin(X["time"] * (2 * np.pi / NUMBER_OF_SECONDS_IN_DAY))
    X["cos_day"] = np.cos(X["time"] * (2 * np.pi / NUMBER_OF_SECONDS_IN_DAY))
    return X


def _get_price_columns(X: pd.DataFrame) -> List[str]:
    """
    Get the columns of the input DataFrame that contain the price data.
    """
    return [col for col in X.columns if "price" in col]


def _get_volume_columns(X: pd.DataFrame) -> List[str]:
    """
    Get the columns of the input DataFrame that contain the volume data.
    """
    return [col for col in X.columns if "volume" in col]


def _get_polynomial_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 2 new polynomial features ("upper_shadow", "lower_shadow")
    and adds these to our DataFrame.
    """
    X["upper_shadow"] = X["high"] - np.maximum(X["close"], X["open"])
    X["lower_shadow"] = np.minimum(X["close"], X["open"]) - X["low"]
    return X


def _get_price_rolling_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the rolling mean of the lagged price values
    over a 24 hour window.
    """
    X["pr_24hr_mavg"] = X[_get_price_columns(X)].agg(["mean"], axis=1)
    X["pr_24hr_mstd"] = X[_get_price_columns(X)].agg(["std"], axis=1)
    return X


def _get_volume_rolling_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the rolling mean of the lagged volume values over
    a 24 hour window.
    """
    X["volume_24hr_mavg"] = X[_get_volume_columns(X)].agg(["mean"], axis=1)
    X["volume_24hr_mstd"] = X[_get_volume_columns(X)].agg(["std"], axis=1)
    return X


def _get_price_percentage_return(X: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Add the price return of the last `hours` to the input DataFrame.
    """
    X[f"percentage_return_{hours}_hour"] = (
        X["price_1_hour_ago"] - X[f"price_{hours}_hour_ago"]
    ) / X[f"price_{hours}_hour_ago"]
    return X


class RSI(BaseEstimator, TransformerMixin):
    """
    Calculates the RSI over the past 14 days from the current price point
    and adds it to our features Dataframe.

    New columns are:
        - 'rsi'

    Thank you Pau Labarta Bajo for figuring out this implementation 🙏
    https://github.com/Paulescu/hands-on-train-and-deploy-ml/blob/main/src/preprocessing.py
    """

    def __init__(self, window: int = 14):
        self.window = window

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None
    ) -> "RSI":
        """
        In this scenario, the fit method isn't doing anything. But it must be
        implemented. This is a scenario of an estimator without parameters.
        """
        return self

    def _add_indicator(self, row: pd.Series) -> float:
        return pd.Series([ta.momentum.rsi(row, window=self.window)[-1]])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the RSI and add it to the input DataFrame."""
        df = X[_get_price_columns(X)].apply(self._add_indicator, axis=1)
        df.columns = ["rsi"]
        X = pd.concat([X, df], axis=1)
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse the log of every cell of the DataFrame."""
        X.drop(columns=["rsi"], inplace=True)
        return X


def get_feature_engineering_pipeline(pp_rsi_window: int = 14) -> Pipeline:
    """Returns the preprocessing pipeline."""
    return make_pipeline(
        # Polynomial features (upper and lower shadows)
        FunctionTransformer(_get_polynomial_features),
        # Windows (moving average, std)
        FunctionTransformer(_get_price_rolling_statistics),
        FunctionTransformer(_get_volume_rolling_statistics),
        # Trends
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 2}),
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 3}),
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 4}),
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 12}),
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 24}),
        # Momentum
        RSI(pp_rsi_window),
    )


@click.command()
@click.option(
    "--data-file-name",
    "-d",
    type=str,
    default="BTC-USD_ohlc_data.parquet",
    show_default=True,
    help="Name of the data file you want to preprocess",
)
@click.option(
    "--window",
    "-w",
    type=int,
    default=24,
    show_default=True,
    help="Window size of hours to generate lag features over",
)
@click.option(
    "--step",
    "-s",
    type=int,
    default=1,
    show_default=True,
    help="Step size to cut window into individual time stamps",
)
@click.option(
    "--track",
    "-t",
    is_flag=True,
    help="Track and version datasets with Weights & Biases",
)
def generate_full_features_and_target_datasets(
    data_file_name: Optional[str] = "BTC-USD_ohlc_data.parquet",
    window: Optional[int] = 24,
    step: Optional[int] = 1,
    track: Optional[bool] = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full workflow for preprocessing the raw crypto currency data
    scraped from the Coinbase Exchange REST API, converting it from
    a time series format to a (features, target) format for ML by generating
    lag features, window features, technical indicator features, and our
    target variable.

    Can be used as a module in another script or as a CLI tool.

    Args:
        data_file_name (Optional[str]): file name of scraped parquet file
        window (Optional[int]): window size of hours to generate lag features over
        steps (Optional[int]): step size to cut window into individual time stamps
        track (Optional[bool]): version datasets with Weights & Biases
        save_local (Optional[bool]): save datasets locally

    Returns:
        pd.DataFrame(X), pd.Series(target)
    """
    # Locate data file
    logger.info(f"Locating {data_file_name}...")
    data_path: Path = DATA_DIR / data_file_name
    if data_path.exists():
        logger.info(f"{data_file_name[0:7]} found at: {data_path} 🟢\n")
    else:
        logger.error("Unable to locate data file 🔴\n")
        raise FileNotFoundError

    # Build features and target datasets
    logger.info(f"Starting preprocessing for: {data_file_name[0:7]} 🚀\n")
    features, target = transform_ts_data_into_lagged_features_and_target(
        path_to_input=data_path,
        window_seq_len=window,
        step_size=step,
    )
    logger.info(f"{data_file_name[0:7]} successfully split into features and target ✨")

    # Build preprocessing pipeline
    logger.info(f"Building feature engineering pipeline for {data_file_name[0:7]} 🔧\n")
    feature_engineering_pipeline = get_feature_engineering_pipeline()

    # Fit preprocessing pipeline
    logger.info("Fitting feature engineering pipeline... 👕\n")
    feature_engineering_pipeline.fit(features)

    # Transform X data
    logger.info("Transforming data ... 🐛 -> 🦋\n")
    X = feature_engineering_pipeline.transform(features)

    logger.info(f"{'⭐'*10} Successfully generated X and y datasets {'⭐'*10}\n")
    logger.info("Features: \n")
    logger.info(f"{X.head()}\n")
    logger.info(f"{X.info()}\n")
    logger.info("Targets: \n")
    logger.info(f"{target.head()}\n")
    logger.info(f"{target.info()}\n")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {target.shape}\n")

    # Save datasets locally
    logger.info("Saving datasets locally...")
    X.to_parquet(
        DATA_DIR / f"{data_file_name[0:7]}_X_full_preprocessed_data.parquet",
        index=True,
    )
    pd.DataFrame(data=target).to_parquet(
        DATA_DIR / f"{data_file_name[0:7]}_y_full_preprocessed_data.parquet",
        index=True,
    )
    logger.info("Datasets saved 🟢")

    # Version datasets with wandb if specified
    if track:
        logger.info("Versioning data @ W&B backend ✨")
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name="post_processed_features_target_datasets",
        )
        X_table = wandb.Table(dataframe=X)
        y_table = wandb.Table(dataframe=pd.DataFrame(data=target))
        run.log(
            {
                f"{data_file_name[0:7]}_X_full_preprocessed_data": X_table,
                f"{data_file_name[0:7]}_y_full_preprocessed_data": y_table,
            }
        )
        wandb.finish()
        logger.info("Datasets successfully versioned 🟢")

    # Return the fully preprocessed features and target dataset
    return X, target


if __name__ == "__main__":
    generate_full_features_and_target_datasets()
