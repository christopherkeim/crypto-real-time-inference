"""
This is a minimum viable web scraber CLI tool that takes utilizes the
Coinbase Exchange REST API to download candles for a given time
window (i.e. 3 years) at a given time resolution (i.e. 1 hour).

For simplicity I will target Bitcoin, which is product id "BTC-USD":

https://api.exchange.coinbase.com/products/{product_id}/
    candles?start={start_day}&end={end_day}&granularity=3600'
"""

from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import requests
import click
import pandas as pd

from src.paths import DATA_DIR
from src.logger import get_console_logger


logger = get_console_logger(name="dataset_generation")


@click.command()
@click.option("--product-id", default="BTC-USD", help="Examples 'BTC-USD', 'ETH-USD'")
@click.option("--from-day", default="2022-09-01", help="Start 'YEAR-MONTH-DAY' string")
@click.option("--to-day", default="2023-09-01", help="End 'YEAR-MONTH-DAY' string")
def download_ohlc_data_from_coinbase(
    product_id: Optional[str] = "BTC-USD",
    from_day: Optional[str] = "2022-09-01",
    to_day: Optional[str] = "2023-09-01",
) -> Path:
    """
    This function downloads historical candles (Open, High, Low, Close, Volume)
    from the Coinbase Exchange REST API and saves them to the `data` directory as
    a parquet file.

    Args:
        product_id (Optional[str]): product_id of the target currency
        from_day (Optional[str]): start day in Year-Month-Day
        to_day (Optional[str]): end day in Year-Month-Day

    Returns:
        Path(DATA_DIR / f"{product_id}_ohlc_data.parquet"): path to full
        OHLC candle download parqet file for crypto currency target

    Reference:
        https://api.exchange.coinbase.com/products/{product_id}/
            candles?start={start_day}&end={end_day}&granularity=3600'
    """

    # Construct a list of days as strings
    days = pd.date_range(start=from_day, end=to_day, freq="1D")
    days = [day.strftime("%Y-%m-%d") for day in days]

    # Create an empty DataFrame
    data = pd.DataFrame()

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

    # Save the full data as a parquet file
    data.to_parquet(DATA_DIR / f"{product_id}_ohlc_data.parquet", index=False)

    return DATA_DIR / f"{product_id}_ohlc_data.parquet"


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
    start = f"{day}T00:00:00"
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    end = f"{end}T00:00:00"

    # Call the Coinbase Exchange REST API for this product, day, and granularity
    URL = f"https://api.exchange.coinbase.com/products/{product_id}/"
    URL += f"candles?start={start}&end={end}&granularity=3600"
    r = requests.get(URL)
    data = r.json()

    # transform list of lists to pandas dataframe and return
    return pd.DataFrame(
        data, columns=["time", "low", "high", "open", "close", "volume"]
    )


if __name__ == "__main__":
    download_ohlc_data_from_coinbase()
