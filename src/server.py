"""
Backend server for inference service.
"""
from typing import Dict
from enum import Enum
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
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
    # Calculate UTC request timestamp
    request_timestamp: str = str(int(datetime.now(timezone.utc).timestamp() * 1000))

    # Compute datetime hour for prediction
    if time_from == "now":
        time_from = datetime.now().strftime("%Y-%m-%dT%H")
    else:
        time_from = (datetime.now() - timedelta(hours=int(time_from))).strftime(
            "%Y-%m-%dT%H"
        )

    # Download the data for that target hour
    raw_data: pd.DataFrame = download_data_for_t_hours(
        product_id=coin,
        date_time_hour=time_from,
        t=24,
    )

    # Engineer features for model consumption
    feature_row: pd.DataFrame = get_feature_row_for_prediction(raw_data, coin)

    # Get the prediction
    price_next_hour: float = predict(feature_row, coin, model_name=model_name)

    # Calculate the predicted difference
    predicted_difference: float = price_next_hour - raw_data["close"].values[0]
    sign: str = "+" if predicted_difference > 0 else ""

    # Calculate UNIX timestamp for time_from
    unix_time_from: str = str(
        int(datetime.strptime(time_from, "%Y-%m-%dT%H").timestamp() * 1000)
    )

    # Construction the Prediction response
    r: PredictionResult = PredictionResult(
        model=model_name,
        coin=coin,
        current_price=f"{raw_data['close'].values[0]:.2f}",
        prediction=f"{price_next_hour:.2f}",
        difference=f"{sign}{predicted_difference:.2f}",
        time=unix_time_from,
        request_timestamp=request_timestamp,
    )

    # Return it as a JSON object
    return {"prediction": r}


if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        workers=1,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
