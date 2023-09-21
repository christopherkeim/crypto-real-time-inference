"""
Training pipeline for deep learning algorithms using Tensorflow and Keras.

Neural network architectures supported:
  - Convolutional Neural Network (CNN)
  - Long Short Term Memory (LSTM) Neural Network

The approach I've taken here is to train base models for production on
a GPU, and then update them within the production environment on small batches of
data every 24 hours using a CPU. 

The recommended model, based on a battery of experiments, is a CNN with the
architecture configurations provided. It is in fact lightweight enough to 
test out on a local machine using a CPU, but do note that training on a GPU
is much faster (and recommended).
"""

from typing import Dict, Tuple, List
from pathlib import Path
import os
import pickle
import wandb
import click

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras import Sequential
from keras.layers import InputLayer, Dense, Activation, Conv1D, Flatten, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from src.logger import get_console_logger
from src.paths import MODELS_DIR, GRAPHS_DIR, DATA_DIR


logger = get_console_logger("neural_net_training")


def get_nn_model_from_name(
    name: str,
    n_features: int,
) -> Tuple[Sequential, Dict]:
    """
    Returns the model constructor for an input name along with
    its hyperparamters.
    """
    from hyperparam_config import CNN_PARAMS, LSTM_PARAMS

    if name == "cnn":
        return build_cnn_model(n_features), CNN_PARAMS
    elif name == "lstm":
        return build_lstm_model(n_features), LSTM_PARAMS
    else:
        raise NotImplementedError(f"{name} not implemented")


def generate_scaled_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    SCALER_PATH: Path = MODELS_DIR / "X_scaler_model.pkl"

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


def evaluate(model: Sequential, X_set: pd.DataFrame, y_set: pd.Series) -> float:
    """
    Evaluates a given model and returns its Mean Absolute Error.
    """
    # Generate predictions for evaluation
    y_preds: np.ndarray = model.predict(X_set).flatten()

    # Evaluate model and store its MAE
    model_mae: float = mean_absolute_error(y_set, y_preds)

    return model_mae


def build_cnn_model(n_features: int) -> Sequential:
    """
    Builds a CNN neural network model.
    """
    # Build the CNN neural network using the Sequential API
    cnn_model = Sequential()
    cnn_model.add(InputLayer((n_features, 1)))
    cnn_model.add(Conv1D(256, kernel_size=4))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(Dense(128, activation="linear"))
    cnn_model.add(Activation("relu"))
    cnn_model.add(Dense(1, activation="linear"))

    # Compile the CNN neural network
    cnn_model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=0.001),
        metrics=["mean_absolute_error"],
    )

    return cnn_model


def build_lstm_model(n_features: int) -> Sequential:
    """
    Builds a LSTM neural network model.
    """
    # Build the LSTM neural network using the Sequential API
    lstm_model = Sequential()
    lstm_model.add(InputLayer((n_features, 1)))
    lstm_model.add(LSTM(256, return_sequences=True))
    lstm_model.add(LSTM(256))
    lstm_model.add(Dense(128, activation="relu"))
    lstm_model.add(Activation("relu"))
    lstm_model.add(Dense(128))
    lstm_model.add(Activation("relu"))
    lstm_model.add(Dense(1, activation="linear"))

    # Compile the LSTM neural network
    lstm_model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=0.001),
        metrics=["mean_absolute_error"],
    )

    return lstm_model


def _save_loss_plot_locally(
    history: Dict,
    model_name: str,
    title: str,
) -> None:
    """
    Plots the learning curve of a neural network gathered from its
    training and saves it locally as a JPEG.
    """
    epochs = range(1, len(history.history["loss"]) + 1)
    history_df = pd.DataFrame()
    history_df["loss"] = history.history["loss"]
    history_df["val_loss"] = history.history["val_loss"]
    history_df["epochs"] = epochs
    history_df = history_df.set_index(["epochs"]).sort_index()[["val_loss", "loss"]]

    # Figure
    plt.figure(figsize=(12, 8))

    # Plot the results
    loss_graph = sns.lineplot(history_df)

    # Titles
    plt.title(f"{title}", fontsize=16)
    plt.ylabel("loss", fontsize=12)
    plt.xlabel("epochs", fontsize=12)

    # Save the graph locally
    picture = loss_graph.get_figure()
    picture.savefig(GRAPHS_DIR / f"{model_name}_{title}.jpeg", format="jpeg")


def _save_predictions_plot_locally(
    model: Sequential,
    X_set: pd.DataFrame,
    y_set: pd.Series,
    model_name: str,
    title: str,
) -> None:
    """
    Plots a given models predictions on an X_set and y_set and saves
    the figure as a .jpeg file locally.
    """

    # Generate predictions and construct the predictions DataFrame
    y_preds: np.ndarray = model.predict(X_set).flatten()
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
    picture.savefig(GRAPHS_DIR / f"{model_name}_{title}.jpeg", format="jpeg")


def _create_wandb_predictions_table(
    run: wandb.run,
    model: Sequential,
    X_set: pd.DataFrame,
    y_set: pd.Series,
    title: str,
) -> None:
    """
    Constructs a graph of a given models predictions on an X_set and y_set
    and logs it to the W&B backend.
    """
    y_preds = model.predict(X_set).flatten()
    pred_df = pd.DataFrame({"predictions": y_preds, "actuals": y_set})
    table = wandb.Table(dataframe=pred_df[-100:])

    run.log({f"{title}": table})


def train_nn_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    learning_rate: int = 0.001,
    epochs: int = 20,
    batch_size: int = 128,
    track: bool = False,
) -> None:
    """
    Trains a specified neural network.
    """

    # Split our datasets into X_train, y_train, X_val, y_val, X_test, y_test
    logger.info("Splitting dataset for modeling ...\n")
    X_train, y_train, X_val, y_val, X_test, y_test = ts_train_val_test_split(
        X, y, train_split=0.8
    )

    # Construct the neural network Sequential model
    nn_model, PARAMS = get_nn_model_from_name(model_name, n_features=X.shape[1])

    # Check for GPU
    if tf.config.list_physical_devices("GPU"):
        logger.info("Training on GPU...")
    else:
        logger.info("Training on CPU...")

    # Define a model checkpoint to save our best models to
    cp = ModelCheckpoint(str(MODELS_DIR / f"{model_name}_model/"), save_best_only=True)

    # Train the neural network model
    logger.info(f"Fitting {model_name} model on training data ... 👕\n")
    history = nn_model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[cp],
    )

    # load the best model from our checkpoint
    best_model = load_model(str(MODELS_DIR / f"{model_name}_model/"))

    # Evaluate model and store its validation MAE and test MAE
    logger.info(f"Evaluating {model_name} model... 🔎")
    val_mae = evaluate(best_model, X_val, y_val)
    test_mae = evaluate(best_model, X_test, y_test)
    logger.info(f"Validation MAE: {val_mae} 🎯")
    logger.info(f"Test MAE: {test_mae} 🎯\n")

    # Save graph of loss learning curve
    logger.info(f"Saving loss learning curve graph to {GRAPHS_DIR} 📉\n")
    _save_loss_plot_locally(history=history, model_name=model_name, title="loss_curve")
    logger.info("Learning curve successfully saved 🟢\n")

    # Save graphs of validation predictions and test predictions
    logger.info(f"Saving validation and test prediction graphs to {GRAPHS_DIR} 📈\n")
    _save_predictions_plot_locally(
        best_model, X_val, y_val, model_name, title="validation_predictions"
    )
    _save_predictions_plot_locally(
        best_model, X_test, y_test, model_name, title="test_predictions"
    )
    logger.info("Graphs successfully saved 🟢\n")

    # Upload model to model registry
    if track:
        logger.info("Versioning model @ W&B backend ✨")
        config: Dict = dict(
            lr=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            architecture=f"{model_name}",
            **PARAMS,
        )
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name="neural_net_training",
            notes=f"Baseline for {model_name}_model",
            tags=["baseline", f"{model_name}_model"],
            config=config,
        )
        model_artifact = wandb.Artifact(f"{model_name}_model", type="model")
        # Upload model
        model_artifact.add_dir(
            local_path=(MODELS_DIR / f"{model_name}_model"), name=f"{model_name}_model"
        )
        # Upload loss curve JPEG
        model_artifact.add_file(GRAPHS_DIR / f"{model_name}_loss_curve.jpeg")
        # Upload test predictions JPEG
        model_artifact.add_file(GRAPHS_DIR / f"{model_name}_test_predictions.jpeg")
        # Upoload validation predictions JPEG
        model_artifact.add_file(
            GRAPHS_DIR / f"{model_name}_validation_predictions.jpeg"
        )
        # Log the version
        run.log_artifact(model_artifact, aliases=["latest", "best"])

        # Validation predictions wandb table
        _create_wandb_predictions_table(
            run=run,
            model=best_model,
            X_set=X_val,
            y_set=y_val,
            title=f"{model_name}_model_validation_predictions_table",
        )
        # Test predictions wandb table
        _create_wandb_predictions_table(
            run=run,
            model=best_model,
            X_set=X_test,
            y_set=y_test,
            title=f"{model_name}_model_test_predictions_table",
        )
        # Log error metrics for validation and test
        run.log(
            {
                f"{model_name}_val_mean_average_error": val_mae,
                f"{model_name}_test_mean_average_error": test_mae,
            }
        )
        wandb.finish()
        logger.info("Model successfully versioned 🟢")


@click.command()
@click.option(
    "--model-name",
    "-m",
    type=str,
    default="cnn",
    show_default=True,
    help="Name of the neural net model you want to train",
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
    "--epochs",
    "-e",
    type=int,
    default=20,
    show_default=True,
    help="Epochs to train neural net",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=128,
    show_default=True,
    help="Number of samples per gradient update",
)
@click.option(
    "--track",
    "-t",
    is_flag=True,
    help="Version model in W&B",
)
def nn_training_pipeline(
    model_name: str = "cnn",
    product_id: str = "BTC-USD",
    epochs: int = 20,
    batch_size: int = 128,
    track: bool = False,
) -> None:
    """
    Full neural network training pipeline.

    Args:
        model_name (str): Name of the neural net model you want to train
        product_id (str): Name product data
        epochs (int): Epochs to train neural net
        batch_size (int): Number of samples per gradient update
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
    X: pd.DataFrame = generate_scaled_features(X)

    # Train the model
    train_nn_model(model_name, X, y, epochs=epochs, batch_size=batch_size, track=track)


if __name__ == "__main__":
    nn_training_pipeline()
