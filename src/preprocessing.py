"""
Prepocessing steps after data has been scraped.
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import click

import pandas as pd
from sktime.transformations.series.lag import Lag
import ta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.paths import DATA_DIR
from src.logger import get_console_logger

logger = get_console_logger("dataset_preprocessing")


def transform_ts_data_into_features_and_target(
    # ts_data: pd.DataFrame,
    path_to_input: Optional[Path] = DATA_DIR / "BTC-USD_ohlc_data.parquet",
    input_seq_len: Optional[int] = 24,
    step_size: Optional[int] = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms the raw data from time-series format into a (features, target)
    format that can be used to train Supervised ML models.
    """
    # load parquet file
    ts_data = pd.read_parquet(path_to_input)
    ts_data["time"] = pd.to_datetime(ts_data["time"], unit="s")
    ts_data.sort_values(by=["time"], inplace=True)
    ts_data = ts_data.drop_duplicates("time", keep="first")
    ts_data.reset_index(drop=True, inplace=True)

    # Define a 1 -> 24 hour lag for our x features
    x_lag = Lag([i for i in reversed(range(1, 25))])

    # Create lagged columns for closing price
    x_price = ts_data[["close"]].values
    x_lagged_price = x_lag.fit_transform(x_price)

    # Create lagged columns for volume
    x_volume = ts_data[["volume"]].values
    x_lagged_volume = x_lag.fit_transform(x_volume)

    # Price lagged features DataFrame
    price_features = pd.DataFrame(
        x_lagged_price,
        columns=[f"price_{i+1}_hour_ago" for i in reversed(range(input_seq_len))],
    )
    # Volume lagged features DataFrame
    volume_features = pd.DataFrame(
        x_lagged_volume,
        columns=[f"volume_{i+1}_hour_ago" for i in reversed(range(input_seq_len))],
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
    full_df["time"] = pd.PeriodIndex(full_df["time"], freq="H")
    full_df = full_df.set_index(["time"]).sort_index()

    # Now extract our features and targets
    features = full_df.drop("target_price_next_hour", axis=1)
    targets = full_df["target_price_next_hour"]

    return features, targets


def _get_price_columns(X: pd.DataFrame) -> List[str]:
    """
    Get the columns of the input DataFrame that contain the price data.
    """
    return [col for col in X.columns if "price" in col]


def _get_volume_columns(X: pd.DataFrame) -> List[str]:
    """
    Get the columns of the input DataFrame that contain the price data.
    """
    return [col for col in X.columns if "volume" in col]


def _get_price_rolling_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the rolling mean of a lagged timepoint's values over
    a given window.
    """
    X["pr_24hr_mavg"] = X[_get_price_columns(X)].agg(["mean"], axis=1)
    X["pr_24hr_mstd"] = X[_get_price_columns(X)].agg(["std"], axis=1)
    return X


def _get_volume_rolling_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the rolling mean of a lagged timepoint's values over
    a given window.
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


def get_preprocessing_pipeline(pp_rsi_window: int = 14) -> Pipeline:
    """Returns the preprocessing pipeline."""
    return make_pipeline(
        # Windows (moving average, std)
        FunctionTransformer(_get_price_rolling_statistics),
        FunctionTransformer(_get_volume_rolling_statistics),
        # Trends
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 2}),
        FunctionTransformer(_get_price_percentage_return, kw_args={"hours": 24}),
        # Momentum
        RSI(pp_rsi_window),
        # select columns
        # FunctionTransformer(get_subset_of_features)
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
    "--save-datasets",
    "-s",
    type=bool,
    default=True,
    show_default=True,
    help="Save datasets locally",
)
def generate_full_features_and_target_datasets(
    data_file_name: Optional[str] = "BTC-USD_ohlc_data.parquet",
    save_datasets: Optional[bool] = True,
) -> None:
    logger.info(f"Locating {data_file_name}...")
    data_path: Path = DATA_DIR / data_file_name
    if data_path.exists():
        logger.info(f"{data_file_name[0:8]} found at: {data_path} 🟢\n")
    else:
        logger.error("Unable to locate data file 🔴\n")
        raise FileNotFoundError

    logger.info(f"Starting preprocessing for: {data_file_name[0:8]} 🚀\n")
    features, target = transform_ts_data_into_features_and_target(
        path_to_input=data_path
    )

    logger.info(f"{data_file_name[0:8]} successfully split into features and target ✨")
    logger.info(f"Building preprocessing pipeline for {data_file_name[0:8]} 🔧\n")
    preprocessing_pipeline = get_preprocessing_pipeline()

    logger.info("Fitting preprocessing pipeline... 👕\n")
    preprocessing_pipeline.fit(features)

    logger.info("Transforming data ... 🐛 -> 🦋\n")
    X = preprocessing_pipeline.transform(features)

    logger.info(f"{'⭐'*10} Successfully generated X and y datasets {'⭐'*10}\n")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {target.shape}\n")
    if save_datasets:
        logger.info("Saving datasets locally...")
        X.to_parquet(
            DATA_DIR / f"{data_file_name[0:8]}_X_full_preprocessed_data.parquet",
            index=True,
        )
        target.to_parquet(
            DATA_DIR / f"{data_file_name[0:8]}_y_full_preprocessed_data.parquet",
            index=True,
        )
        logger.info("Datasets saved 🟢")


if __name__ == "__main__":
    generate_full_features_and_target_datasets()
