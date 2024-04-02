import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--test_data", type=str, help="Path to the test data")
    args = parser.parse_args()

    credential = DefaultAzureCredential()
    try:
        ml_client = MLClient.from_config(credential, path="config.json")

    except Exception as ex:
        print("HERE IN THE EXCEPTION BLOCK")
        print(ex)

    version = "v" + time.strftime("%Y.%m.%d.%H%M%S", time.gmtime())
    train_data = Data(
        name="TrainData",
        description="Training data",
        path=args.train_data,
        type=AssetTypes.MLTABLE,
        version=version,
    )
    ml_client.data.create_or_update(train_data)
    test_data = Data(
        name="TestData",
        description="Test data",
        path=args.test_data,
        type=AssetTypes.MLTABLE,
        version=version,
    )
    ml_client.data.create_or_update(test_data)


if __name__ == "__main__":
    main()
