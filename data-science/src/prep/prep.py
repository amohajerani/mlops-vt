"""
Prepares raw data and provides training and test datasets
"""

import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow

TARGET_COL = "VISIT_TIME"
# set a random seed for numpy
np.random.seed(42)


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument(
        "--table_name",
        type=str,
        default="mlmonitoring",
        help="Table name in ADX for logging",
    )

    args = parser.parse_args()

    return args


def log_training_data(df, table_name):
    from obs.collector import Online_Collector

    collector = Online_Collector(table_name)
    collector.batch_collect(df)


def main(args):
    """Read, split, and save datasets"""

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    data = pd.read_csv((Path(args.raw_data)))

    # ------------ Data Cleaning ------------ #
    # remove rows with missing values for the target column
    data = data.dropna(subset=[TARGET_COL])

    # ------------- Split Data ------------- #
    # -------------------------------------- #

    # Split data into train and test datasets

    random_data = np.random.rand(len(data))

    msk_train = random_data < 0.8
    msk_test = random_data >= 0.8

    train = data[msk_train]
    test = data[msk_test]

    mlflow.log_metric("train size", train.shape[0])
    mlflow.log_metric("test size", test.shape[0])

    train.to_parquet((Path(args.train_data) / "train.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
