"""
Configurations for model hyperparameters.
"""

LASSO_CV_CONFIG = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

LGBM_CV_CONFIG = {
    "n_estimators": [50, 100, 150, 200],
    "num_leaves": [10, 20, 31, 40, 50],
    "max_depth": [-1, 2, 4, 8, 10],
}

CNN_PARAMS = {
    "CONV1D_1": {"units": 256, "kernel_size": 4},
    "FLATTEN_1": True,
    "DENSE_1": {"units": 128, "activation": "relu"},
    "ACTIVATION_1": "relu",
    "DENSE_2": {"units": 128, "activation": "linear"},
    "ACTIVATION_2": "relu",
    "OUTPUT": {"units": 1, "activation": "linear"},
}

LSTM_PARAMS = {
    "LSTM_1": {"units": 256, "return_sequences": True},
    "LSTM_2": {"units": 256},
    "DENSE_1": {"units": 128, "activation": "relu"},
    "ACTIVATION_1": "relu",
    "DENSE_2": {"units": 128, "activation": "linear"},
    "ACTIVATION_2": "relu",
    "OUTPUT": {"units": 1, "activation": "linear"},
}
