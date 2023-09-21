"""
This is a minimum viable web scraber CLI tool that utilizes the
Coinbase Exchange REST API to download candles for a given time
window (i.e. 3 years) at a given time resolution (i.e. 1 hour).

For simplicity I will target Bitcoin, which is product id "BTC-USD":

https://api.exchange.coinbase.com/products/{product_id}/
    candles?start={start_day}&end={end_day}&granularity=3600'
"""

from typing import List, Dict
from datetime import datetime, timedelta
import time
import requests
import click
import pandas as pd

from src.paths import DATA_DIR
from src.logger import get_console_logger


logger = get_console_logger(name="dataset_generation")


@click.command()
@click.option(
    "--product-ids",
    "-p",
    type=str,
    default="BTC-USD",
    show_default=True,
    help="Examples 'BTC-USD ETH-USD'",
)
@click.option(
    "--from-day",
    "-f",
    type=str,
    default="2020-09-01",
    show_default=True,
    help="Start 'YEAR-MONTH-DAY'",
)
@click.option(
    "--to-day",
    "-t",
    type=str,
    default="now",
    show_default=True,
    help="End 'YEAR-MONTH-DAY'",
)
def download_ohlc_data_from_coinbase(
    product_ids: str = "BTC-USD",
    from_day: str = "2020-09-01",
    to_day: str = "now",
) -> None:
    """
    This function downloads historical candles (Open, High, Low, Close, Volume)
    from the Coinbase Exchange REST API and saves them to the `data` directory as
    a parquet file.

    Args:
        product_id (str): product_ids of the target currencies
        from_day (str): start day in Year-Month-Day
        to_day (str): "now" or end day in Year-Month-Day

    Returns:
        None

    Reference:
        https://api.exchange.coinbase.com/products/{product_id}/
            candles?start={start_day}&end={end_day}&granularity=3600'
    """
    # Validation for product_ids string inputs
    if not isinstance(product_ids, str):
        raise TypeError(
            "product_ids must be a string containing product ids separated by spaces."
        )

    # Extract the current day as a date time if "now"
    if to_day == "now":
        to_day = datetime.now().strftime("%Y-%m-%d")

    # Construct a list of days as strings
    days: pd.DatetimeIndex = pd.date_range(start=from_day, end=to_day, freq="1D")
    days: List[str] = [day.strftime("%Y-%m-%d") for day in days]

    # Create an empty DataFrame
    data: pd.DataFrame = pd.DataFrame()

    # START TIME
    start_time: float = time.time()

    meta_data: Dict[str, tuple[int, int]] = {}

    product_ids: List[str] = product_ids.split()

    for product_id in product_ids:
        # Create a download dir for this currency if it doesn't exist
        if not (DATA_DIR / f"{product_id}_downloads").exists():
            logger.info(f"Creating {product_id} directory for downloads")
            (DATA_DIR / f"{product_id}_downloads").mkdir(parents=True)

        for day in days:
            # Download the file if it doesn't exist
            file_name = DATA_DIR / f"{product_id}_downloads" / f"{day}.parquet"
            if file_name.exists():
                logger.info(f"File {file_name} already exists, skipping")
                data_one_day = pd.read_parquet(file_name)
            else:
                logger.info(f"Downloading {product_id} data for {day}")
                data_one_day = _download_data_for_one_day(product_id, day)
                data_one_day.to_parquet(file_name, index=False)

            # Combine current day's file with the rest of the data
            data = pd.concat([data, data_one_day])

        # Save the shape of this dataset to our meta data
        meta_data[product_id] = data.shape

        # Save the full data as a parquet file
        data.to_parquet(DATA_DIR / f"{product_id}_ohlc_data.parquet", index=False)

    # END TIMER
    end_time: float = time.time() - start_time

    logger.info(f"{'⭐'*10} Downloaded OHLC Candles for {'⭐'*10}")
    logger.info("Crypto Currencies 💲: ")
    for key in meta_data:
        logger.info(f"\t🪙 {key}: ({meta_data[key]})")
    logger.info(f"START DAY: {from_day}")
    logger.info(f"END DAY: {to_day}")
    logger.info(f"Completed in: {end_time} seconds")


def _download_data_for_one_day(product_id: str, day: str) -> pd.DataFrame:
    """
    Downloads a candle for one day and returns a pandas Dataframe.

    Args:
        product_id (str): product_id of the target currency
        day (str): day in Year-Month-Day

    Returns:
        pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    """

    # Create the start and end date strings
    start: str = f"{day}T00:00:00"
    end: str = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    end: str = f"{end}T00:00:00"

    # Call the Coinbase Exchange REST API for this product, day, and granularity
    URL: str = f"https://api.exchange.coinbase.com/products/{product_id}/"
    URL += f"candles?start={start}&end={end}&granularity=3600"
    r: requests.models.Response = requests.get(URL)
    data: List[List[int, float, float, float, float, float]] = r.json()

    # Transform list of lists to pandas dataframe and return
    return pd.DataFrame(
        data, columns=["time", "low", "high", "open", "close", "volume"]
    )


if __name__ == "__main__":
    download_ohlc_data_from_coinbase()
