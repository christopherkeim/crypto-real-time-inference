from typing import List, Dict
import click
import os
import wandb


from src.logger import get_console_logger
from src.paths import MODELS_DIR


logger = get_console_logger("model_downloads")


def download_model_from_wandb(model_name: str) -> None:
    """
    Fetches and downloads a specified model from the W&B backend.
    """
    logger.info(f"Fetching {model_name}_model from W&B backend ✨")

    run = wandb.init()
    # Fetch the model using the Public API
    reg_model = wandb.use_artifact(
        f"{os.environ['WANDB_ENTITY']}/model-registry/{model_name}_model:latest",
        type="model",
    )
    # Download the model to disk
    model_path = reg_model.download(MODELS_DIR)
    run.finish()

    logger.info(f"Successfully downloaded model to: {model_path} 🟢")


@click.command()
@click.option(
    "--selection",
    "-s",
    type=str,
    default="all",
    show_default=True,
    help="'all' 'nn', or 'ml'",
)
def download_latest_models_from_wandb(selection: str = "all") -> None:
    """
    Takes a selection value ("all", "nn", or "ml") and downloads the latest
    model versions from W&B.
    """
    MODELS: Dict[str, List[str]] = {
        "all": ["cnn", "lasso", "X_scaler"],
        "nn": ["cnn", "X_Scaler"],
        "ml": ["lasso", "X_scaler"],
    }

    logger.info(f"Beginning download process for model selection: {selection} 🤖")

    for model in MODELS[selection]:
        download_model_from_wandb(model)

    logger.info("Models successfully downloaded 🟢")


if __name__ == "__main__":
    download_latest_models_from_wandb()
