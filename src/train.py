"""
Training pipeline for machine learning algorithms.

Models supported:
  - Lasso Regressor
  - LightGBM Regressor

NOTE: Hyperparameter tuning is not fully implemented. In a larger 
scale project, we would likely use W&B Sweeps along with a config.yaml 
file to perform exhaustive optimization. 

On a local machine it is computationally expensive to run a grid search cross-
validation, and not recommended. Manual experiments confirmed that default 
parameters for these models are optimal. However, I will keep this in for future 
projects.
"""

from typing import Dict, Tuple, List, Callable
from pathlib import Path
import os
import pickle
import wandb
import click

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor

from src.logger import get_console_logger
from src.paths import MODELS_DIR, DATA_DIR, GRAPHS_DIR


logger = get_console_logger("model_training")


def get_model_constructor_from_name(name: str) -> Callable:
    """
    Returns the model constructor for an input name.
    """
    if name == "lasso":
        return Lasso
    elif name == "lgbm":
        return LGBMRegressor
    else:
        raise NotImplementedError(f"{name} not implemented")


def generate_scaled_features(X: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    SCALER_PATH: Path = MODELS_DIR / f"{product_id}_X_scaler_model.pkl"

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


def ts_train_val_test_split(
    X: pd.DataFrame, y: pd.Series, train_split: float = 0.8
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Takes a set of features X and a target y and splits it into X_train, y_train,
    X_val, y_val, and X_test, y_test datasets according to a given train_split
    percentage.
    """
    # train_split must be a float between 0.1 and 0.9
    if not isinstance(train_split, float):
        raise TypeError("train_split must be a float between 0.1 and 0.9")

    if train_split < 0.1 or train_split >= 1.0:
        raise ValueError("train_split must be a float between 0.1 and 0.9")

    # Calculate sample slice lengths for train, val, and test
    train_sample_size: int = int(train_split * len(X))
    val_split: float = (1.0 - train_split) / 2
    val_sample_size: int = int((train_split + val_split) * len(X))

    # Split the datasets
    X_train, y_train = X[:train_sample_size], y[:train_sample_size]
    X_val, y_val = (
        X[train_sample_size:val_sample_size],
        y[train_sample_size:val_sample_size],
    )
    X_test, y_test = X[val_sample_size:], y[val_sample_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_hyperparameters(
    model_name: str, model: Callable, X_train: pd.DataFrame, y_train: pd.Series
) -> Dict:
    """
    Performs Grid Search Cross-Validation on a given model.
    """
    from sklearn.model_selection import GridSearchCV
    from hyperparam_config import LASSO_CV_CONFIG, LGBM_CV_CONFIG

    if model_name == "lasso":
        params = LASSO_CV_CONFIG

    elif model_name == "lgbm":
        params = LGBM_CV_CONFIG

    grid_search_cv = GridSearchCV(
        estimator=model(),
        param_grid=params,
        cv=5,
        verbose=1,
    )

    grid_search_cv.fit(X_train, y_train)

    best_params: Dict[str, int] = grid_search_cv.best_params_

    return best_params


def evaluate(model: Callable, X_set: pd.DataFrame, y_set: pd.Series) -> float:
    """
    Evaluates a given model and returns its Mean Absolute Error.
    """
    # Generate predictions for evaluation
    y_preds: np.ndarray = model.predict(X_set)

    # Evaluate model and store its MAE
    model_mae: float = mean_absolute_error(y_set, y_preds)

    return model_mae


def _save_predictions_plot_locally(
    model: Callable, X_set: pd.DataFrame, y_set: pd.Series, title: str
) -> None:
    """
    Plots a given models predictions on an X_set and y_set and saves
    the figure as a .jpeg file locally.
    """
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Generate predictions and construct the predictions DataFrame
    y_preds: np.ndarray = model.predict(X_set)
    pred_df = pd.DataFrame({"predictions": y_preds, "actuals": y_set})

    # Figure
    plt.figure(figsize=(12, 8))

    # Plot the results
    preds_graph = sns.lineplot(pred_df[-100:])

    # Titles
    plt.title(f"{title}", fontsize=16)
    plt.ylabel("price_next_hour", fontsize=12)
    plt.xlabel("time", fontsize=12)

    # Save the graph locally
    picture = preds_graph.get_figure()
    picture.savefig(GRAPHS_DIR / f"{title}.jpeg", format="jpeg")


def _create_wandb_predictions_table(
    run: wandb.run,
    model: Callable,
    X_set: pd.DataFrame,
    y_set: pd.Series,
    title: str,
) -> None:
    """
    Constructs a graph of a given models predictions on an X_set and y_set
    and logs it to the W&B backend.
    """
    y_preds = model.predict(X_set)
    pred_df = pd.DataFrame({"predictions": y_preds, "actuals": y_set})
    table = wandb.Table(dataframe=pred_df[-100:])

    run.log({f"{title}": table})


def train(
    model_name: str,
    product_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    tune_hyperparameters: bool = False,
    track: bool = False,
) -> None:
    """
    Trains a model, tunes hyperparameters if specified, saves model artifact
    to disk, and versions it in model registry if specified.

    Args:
        X (pd.DataFrame): Features dataset
        y (pd.Series): target dataset
        model_name (str): Name of the model you want to train
        tune_hyperparams (bool): Tune model hyperparameters
        track (bool): Version model in W&B

    Returns:
        None
    """

    # Split our datasets into X_train, y_train, X_val, y_val, X_test, y_test
    logger.info("Splitting dataset for modeling ...\n")
    X_train, y_train, X_val, y_val, X_test, y_test = ts_train_val_test_split(
        X, y, train_split=0.8
    )

    # Convert the model name string -> model constructor (Callable)
    model_constructor: Callable = get_model_constructor_from_name(model_name)

    # Hyperparameter optimization
    if tune_hyperparameters:
        # Run grid search optimization to find the best hyperparameters
        logger.info(f"Optimizing parameters for {model_name} 🕒\n")
        best_params = find_best_hyperparameters(
            model_name, model_constructor, X_train, y_train
        )

        logger.info(f"Optimized paramters for {model_name} found: ")
        logger.info(f"{best_params}\n")

        # Instantiate the model with the optimized hyperparameters
        model = model_constructor(**best_params)
        model_is_best: bool = True
        config = best_params

    # Default hyperparameters
    else:
        logger.info("Using default parameters 🔧\n")
        model = model_constructor()
        model_is_best: bool = False
        config = {"default": True}

    # Train the model
    logger.info(f"Fitting {model_name} model on {product_id} training data ... 👕\n")
    model.fit(X_train, y_train)

    # Evaluate model and store its validation MAE and test MAE
    logger.info(f"Evaluating {product_id}_{model_name} model... 🔎")
    val_mae = evaluate(model, X_val, y_val)
    test_mae = evaluate(model, X_test, y_test)
    logger.info(f"Validation MAE: {val_mae} 🎯")
    logger.info(f"Test MAE: {test_mae} 🎯\n")

    # Save model to disk
    logger.info(f"Saving {product_id}_{model_name} model to {MODELS_DIR}\n")
    with open(MODELS_DIR / f"{product_id}_{model_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"{product_id}_{model_name} model successfully saved to: ")
    logger.info(f"\t {MODELS_DIR} / {product_id}_{model_name}_model.pkl\n")

    # Save graphs of validation predictions and test predictions
    logger.info(f"Saving validation and test prediction graphs to {GRAPHS_DIR} 📈\n")
    _save_predictions_plot_locally(
        model,
        X_val,
        y_val,
        title=f"{product_id}_{model_name}_validation_predictions",
    )
    _save_predictions_plot_locally(
        model,
        X_test,
        y_test,
        title=f"{product_id}_{model_name}_test_predictions",
    )
    logger.info("Graphs successfully saved 🟢\n")

    # Upload model to model registry
    if track:
        logger.info("Versioning model @ W&B backend ✨")
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name="model_training",
            notes=f"Baseline for {product_id}_{model_name}_model",
            tags=["baseline", f"{product_id}_{model_name}_model"],
            config=config,
        )
        model_artifact = wandb.Artifact(
            f"{product_id}_{model_name}_model", type="model"
        )
        # Upload model
        model_artifact.add_file(
            MODELS_DIR / f"{product_id}_{model_name}_model.pkl",
            f"{product_id}_{model_name}_model.pkl",
        )
        # Upload test predictions JPEG
        model_artifact.add_file(
            GRAPHS_DIR / f"{product_id}_{model_name}_test_predictions.jpeg"
        )
        # Upoload validation predictions JPEG
        model_artifact.add_file(
            GRAPHS_DIR / f"{product_id}_{model_name}_validation_predictions.jpeg"
        )
        # Log the version
        if model_is_best:
            run.log_artifact(model_artifact, aliases=["latest", "best"])
        else:
            run.log_artifact(model_artifact)
        # Validation predictions wandb table
        _create_wandb_predictions_table(
            run=run,
            model=model,
            X_set=X_val,
            y_set=y_val,
            title=f"{product_id}_{model_name}_model_validation_predictions_table",
        )
        # Test predictions wandb table
        _create_wandb_predictions_table(
            run=run,
            model=model,
            X_set=X_test,
            y_set=y_test,
            title=f"{product_id}_{model_name}_model_test_predictions_table",
        )
        # Log error metrics for validation and test
        run.log(
            {
                f"{product_id}_{model_name}_val_mean_average_error": val_mae,
                f"{product_id}_{model_name}_test_mean_average_error": test_mae,
            }
        )
        wandb.finish()
        logger.info("Model successfully versioned 🟢")


@click.command()
@click.option(
    "--model-name",
    "-m",
    type=str,
    default="lasso",
    show_default=True,
    help="Name of the model you want to train",
)
@click.option(
    "--product-id",
    "-p",
    type=str,
    default="BTC-USD",
    show_default=True,
    help="Name product data",
)
@click.option(
    "--tune-hyperparams",
    "-h",
    is_flag=True,
    help="Tune model hyperparameters (computationally expensive)",
)
@click.option(
    "--track",
    "-t",
    is_flag=True,
    help="Version model in W&B",
)
def training_pipeline(
    model_name: str = "lasso",
    product_id: str = "BTC-USD",
    tune_hyperparams: bool = False,
    track: bool = False,
) -> None:
    """
    The full model training pipeline.

    Args:
        model_name (str): Name of the model you want to train
        product_id (str): Name product data
        tune_hyperparams (bool): Tune model hyperparameters
        track (bool): Version model in W&B

    Returns:
        None
    """
    # Load the preprocessed X and y datasets into memory
    X_PATH: Path = DATA_DIR / f"{product_id}_X_full_preprocessed_data.parquet"
    Y_PATH: Path = DATA_DIR / f"{product_id}_y_full_preprocessed_data.parquet"

    if not X_PATH.exists():
        logger.error("Unable to locate X data file 🔴\n")
        raise FileNotFoundError

    if not Y_PATH.exists():
        logger.error("Unable to locate y data file 🔴\n")
        raise FileNotFoundError

    logger.info(f"{product_id} preprocessed X data found at: {X_PATH} 🟢\n")
    X: pd.DataFrame = pd.read_parquet(X_PATH)

    logger.info(f"{product_id} preprocessed y data found at: {Y_PATH} 🟢\n")
    y: pd.Series = pd.read_parquet(Y_PATH)["target_price_next_hour"]

    # Scale our X dataset
    logger.info("Scaling X features ... 👉 👈\n")
    X: pd.DataFrame = generate_scaled_features(X, product_id)

    # Train the model
    train(
        model_name, product_id, X, y, tune_hyperparameters=tune_hyperparams, track=track
    )


if __name__ == "__main__":
    training_pipeline()
