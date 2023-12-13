"""
Backend server for inference service.
"""
from typing import Dict, List
from enum import Enum
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi import FastAPI
import uvicorn
import pandas as pd


from src.predict import (
    download_data_for_t_hours,
    get_feature_row_for_prediction,
    predict,
)
from src.logger import get_console_logger


logger = get_console_logger("prediction_server")


app = FastAPI()


class PredictionResult(BaseModel):
    model: str
    coin: str
    current_price: str
    prediction: str
    difference: str
    time: str
    request_timestamp: str
    past_24_hour_prices: List[float]


class Coin(str, Enum):
    bitcoin = "BTC-USD"
    ethereum = "ETH-USD"


class Time(str, Enum):
    now = "now"
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"
    six = "6"
    seven = "7"
    eight = "8"
    nine = "9"


class ModelName(str, Enum):
    cnn = "cnn"
    lasso = "lasso"


@app.get("/api/predict")
def get_prediction(
    coin: Coin = "BTC-USD",
    time_from: Time = "now",
    model_name: ModelName = "cnn",
) -> Dict[str, PredictionResult]:
    """
    Takes a cryptocurrency product id, target hour and model name and
    returns that model's prediction for the cryptocurrency's price point the
    next hour.

    Args:
      coin (str): The product ID of the cryptocurrency
      time_from (str): The datetime hour to predict price point in +1 hour from
      model_name (str): Name of model
    """
    # Get current time
    initial_request_time: datetime = datetime.now()

    # Compute datetime hour for prediction
    prediction_time: datetime = initial_request_time

    if time_from != "now":
        prediction_time -= timedelta(hours=int(time_from))

    # Calculate UTC request timestamp
    request_timestamp: str = str(int(initial_request_time.timestamp() * 1000))

    # Download current candle and past 25 hours for target time
    coinbase_candles: List[List[int | float]] = download_data_for_t_hours(
        product_id=coin,
        date_time_hour=prediction_time,
        t=26,
    )

    # Transform list of lists to Pandas DataFrame
    coinbase_candles_df: pd.DataFrame = pd.DataFrame(
        coinbase_candles, columns=["time", "low", "high", "open", "close", "volume"]
    )

    # Engineer features for model consumption
    feature_row: pd.DataFrame = get_feature_row_for_prediction(
        coinbase_candles_df, coin
    )

    # Extract the current price
    current_price: float = coinbase_candles_df["close"].values[-1]

    # Get the prediction
    price_next_hour: float = predict(feature_row, coin, model_name=model_name)

    # Calculate the predicted difference
    predicted_difference: float = price_next_hour - current_price
    sign: str = "+" if predicted_difference > 0 else ""

    # Calculate UNIX timestamp for time_from and add 1 hour
    unix_time_from: str = str(
        int(
            (
                prediction_time.replace(microsecond=0, second=0, minute=0)
                + timedelta(hours=1)
            ).timestamp()
            * 1000
        )
    )

    # Construct the Prediction response
    response: PredictionResult = PredictionResult(
        model=model_name,
        coin=coin,
        current_price=f"{coinbase_candles_df['close'].values[-1]:.2f}",
        prediction=f"{price_next_hour:.2f}",
        difference=f"{sign}{predicted_difference:.2f}",
        time=unix_time_from,
        request_timestamp=request_timestamp,
        past_24_hour_prices=coinbase_candles_df["close"].to_list()[2:],
    )

    # Return it as a JSON object
    return {"prediction": response}


if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        workers=1,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
