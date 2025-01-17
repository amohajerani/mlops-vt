"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
import numpy

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("azureml.training.tabular")

TARGET_COL = "VISIT_TIME"
# Define your categorical and numerical columns
categorical_features = ["EMPLOYEETYPENAME", "DEGREE"]
numerical_features = [
    "VISIT_TIME_MEAN",
    "GENDERID",
    "PROVIDERAGE",
    "TENURE",
    "VISIT_COUNT",
]


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    args = parser.parse_args()

    return args


class ArrayToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.column_names)
        return X


def model_definition():
    algorithm = XGBRegressor()

    return algorithm


# Create the transformers
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Combine the transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


def build_model_pipeline():
    """
    Defines the scikit-learn pipeline steps.
    """

    column_names = numerical_features + categorical_features
    logger.info("Running build_model_pipeline")
    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            ("array_to_df", ArrayToDataFrameTransformer(column_names)),
            ("preprocessor", preprocessor),
            ("classifier", model_definition()),
        ]
    )
    return pipeline


def main(args):
    """Read train dataset, train model, save trained model"""

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Reorder columns
    column_order = [
        "PROVIDERAGE",
        "TENURE",
        "DEGREE",
        "EMPLOYEETYPENAME",
        "VISIT_TIME_MEAN",
        "VISIT_COUNT",
        "GENDERID",
    ] + [TARGET_COL]
    train_data = train_data[column_order]

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data.drop(columns=[TARGET_COL])
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    model = model_pipeline.fit(X_train, y_train)

    # log model hyperparameters
    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param(
        "n_estimators", model.named_steps["classifier"].get_params()["n_estimators"]
    )
    mlflow.log_param(
        "max_depth", model.named_steps["classifier"].get_params()["max_depth"]
    )
    mlflow.log_param(
        "objective", model.named_steps["classifier"].get_params()["objective"]
    )
    mlflow.log_param(
        "reg_alpha", model.named_steps["classifier"].get_params()["reg_alpha"]
    )
    mlflow.log_param(
        "reg_lambda", model.named_steps["classifier"].get_params()["reg_lambda"]
    )

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)

    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train, color="black")
    plt.plot(y_train, y_train, color="blue", linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    main(args)

    mlflow.end_run()
