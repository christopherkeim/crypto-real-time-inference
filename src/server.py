"""
Backend server for inference service.
"""
from typing import Dict
from enum import Enum
from datetime import datetime
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


class Coin(str, Enum):
    bitcoin = "BTC-USD"


class Time(str, Enum):
    now = "now"


class ModelName(str, Enum):
    cnn = "cnn"
    lasso = "lasso"


@app.get("/api/predict")
def get_prediction(
    coin: Coin = "BTC-USD",
    time_from: Time = "now",
    model_name: ModelName = "cnn",
) -> Dict[str, float]:
    """
    Takes a cryptocurrency product id, target hour and model name and
    returns that model's prediction for the cryptocurrency's price point the
    next hour.

    Args:
      coin (str): The product ID of the cryptocurrency
      time_from (str): The datetime hour to predict price point in +1 hour from
      model_name (str): Name of model
    """

    if time_from == "now":
        time_from = datetime.now().strftime("%Y-%m-%dT%H")

    # Download the data for that target hour
    raw_data: pd.DataFrame = download_data_for_t_hours(
        product_id=coin,
        date_time_hour=time_from,
        t=24,
    )

    # Engineer features for model consumption
    feature_row: pd.DataFrame = get_feature_row_for_prediction(raw_data)

    # Get the prediction
    price_next_hour: float = predict(feature_row, model_name=model_name)

    # Return it as a JSON object
    return {"price_next_hour": price_next_hour}


if __name__ == "__main__":
    uvicorn.run(
        "src.server:app",
        workers=1,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
