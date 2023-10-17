from typing import List, Dict
import click
import os
import wandb


from src.logger import get_console_logger
from src.paths import MODELS_DIR


logger = get_console_logger("model_downloads")


def download_model_from_wandb(product_id: str, model_name: str) -> None:
    """
    Fetches and downloads a specified model from the W&B backend.
    """
    logger.info(f"Fetching {product_id}_{model_name}_model from W&B backend ✨")

    run = wandb.init()
    # Fetch the model using the Public API
    model_url: str = f"{os.environ['WANDB_ENTITY']}/model-registry/"
    model_url += f"{product_id}_{model_name}_model:latest"

    reg_model = wandb.use_artifact(
        model_url,
        type="model",
    )
    # Download the model to disk
    model_path = reg_model.download(MODELS_DIR)
    run.finish()

    logger.info(f"Successfully downloaded model to: {model_path} 🟢")


@click.command()
@click.option(
    "--coin-selection",
    "-c",
    type=str,
    default="all",
    show_default=True,
    help="'all' 'bitcoin', or 'ethereum'",
)
@click.option(
    "--model-selection",
    "-m",
    type=str,
    default="all",
    show_default=True,
    help="'all' 'nn', or 'ml'",
)
def download_latest_models_from_wandb(
    coin_selection: str = "all",
    model_selection: str = "all",
) -> None:
    """
    Takes a coin selection value ("all", "bitcoin", "ethereum") and a model
    selection value ("all", "nn", or "ml") and downloads the latest model
    versions from W&B.
    """
    COINS: Dict[str, List[str]] = {
        "all": ["BTC-USD", "ETH-USD"],
        "bitcoin": ["BTC-USD"],
        "ethereum": ["ETH-USD"],
    }

    MODELS: Dict[str, List[str]] = {
        "all": ["cnn", "lasso", "X_scaler"],
        "nn": ["cnn", "X_Scaler"],
        "ml": ["lasso", "X_scaler"],
    }

    logger.info(f"Beginning download process for model selection: {model_selection} 🤖")
    logger.info(f"and coin selection: {coin_selection} 🪙\n")

    for coin in COINS[coin_selection]:
        for model in MODELS[model_selection]:
            download_model_from_wandb(coin, model)

    logger.info("Models successfully downloaded 🟢")


if __name__ == "__main__":
    download_latest_models_from_wandb()
