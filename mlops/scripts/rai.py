import json
import argparse
import os

from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.entities import Data, PipelineJob
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import dsl, Input
import pandas as pd
import mltable


import time

compute_name = "rai-instance"
experiment_name = "rai-visit-time"
model_name = "vt-model"
rai_scorecard_config_path = "../../rai_scorecard_config.json"
ml_client_registry = None
model_id = None


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    print("Pipeline job can be accessed in the following URL:")

    while created_job.status not in [
        "Completed",
        "Failed",
        "Canceled",
        "NotResponding",
    ]:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))
    assert created_job.status == "Completed"
    return created_job


@dsl.pipeline(
    compute=compute_name,
    description="RAI Pipeline for Visit Time Prediction",
    experiment_name=experiment_name,
)
def rai_regression_pipeline(
    target_column_name,
    train_data,
    test_data,
    score_card_config_path,
):

    ################################
    # Get the RAI components  needed for the RAI pipeline
    ################################
    version = "0.13.0"

    rai_constructor_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_insight_constructor", version=version
    )

    rai_explanation_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_explanation", version=version
    )

    rai_counterfactual_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_counterfactual", version=version
    )

    rai_erroranalysis_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_erroranalysis", version=version
    )

    rai_gather_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_insight_gather", version=version
    )

    rai_scorecard_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_score_card", version=version
    )
    # Initiate the RAIInsights
    model_path = f"azureml:{model_id}"
    print("Model path: ", model_path)
    create_rai_job = rai_constructor_component(
        title="RAI Dashboard for Visit Time Prediction",
        task_type="regression",
        model_info=model_id,
        model_input=Input(
            type="mlflow_model",
            # path=model_path,
        ),
        train_dataset=train_data,
        test_dataset=test_data,
        target_column_name=target_column_name,
        # If your model has extra dependencies, and your Responsible AI job failed to
        # load mlflow model with ValueError, try set use_model_dependency to True.
        # If you have further questions, contact askamlrai@microsoft.com
        use_model_dependency=True,
    )
    create_rai_job.set_limits(timeout=7200)

    # Add an explanation
    explain_job = rai_explanation_component(
        comment="Explanation for the diabetes regression dataset",
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    explain_job.set_limits(timeout=7200)

    # Add error analysis
    erroranalysis_job = rai_erroranalysis_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    erroranalysis_job.set_limits(timeout=7200)

    # Add counterfactual analysis
    counterfactual_job = rai_counterfactual_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        total_cfs=20,
        desired_range=json.dumps([50, 120]),
    )
    counterfactual_job.set_limits(timeout=7200)

    # Combine everything
    rai_gather_job = rai_gather_component(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_1=explain_job.outputs.explanation,
        insight_3=counterfactual_job.outputs.counterfactual,
        insight_4=erroranalysis_job.outputs.error_analysis,
    )
    rai_gather_job.set_limits(timeout=7200)

    rai_gather_job.outputs.dashboard.mode = "upload"
    rai_gather_job.outputs.ux_json.mode = "upload"

    # Generate score card in pdf format for a summary report on model performance,
    # and observe distrbution of error between prediction vs ground truth.
    rai_scorecard_job = rai_scorecard_component(
        dashboard=rai_gather_job.outputs.dashboard,
        pdf_generation_config=score_card_config_path,
    )

    return {
        "dashboard": rai_gather_job.outputs.dashboard,
        "ux_json": rai_gather_job.outputs.ux_json,
        "scorecard": rai_scorecard_job.outputs.scorecard,
    }


def create_rai_datasets(ml_client):
    """
    create a train and a test dataset for the RAI pipeline
    """
    dataset_df = pd.read_csv("train.csv")

    # Split the dataset_df into train and test dataframes. The test should be no more than 5000 rows
    test_size = 5000
    train_size = len(dataset_df) - test_size
    if train_size < 1:
        raise ValueError("Dataset is too small to split into train and test sets")
    # shuffle the dataset before splitting
    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
    train_df = dataset_df.iloc[:train_size]
    test_df = dataset_df.iloc[train_size:]

    # Store the train dataframe as a parquet file
    os.makedirs("rai_train", exist_ok=True)
    train_df.to_parquet("rai_train/train.parquet")
    # Store the test dataframe as a parquet file
    os.makedirs("rai_test", exist_ok=True)
    test_df.to_parquet("rai_test/test.parquet")

    # store the train and test parquet files as mltable
    train_table = mltable.from_parquet_files(
        [{"file": "rai_train/train.parquet"}], include_path_column=False
    )
    test_table = mltable.from_parquet_files(
        [{"file": "rai_test/test.parquet"}], include_path_column=False
    )
    train_table.save("rai_train")
    test_table.save("rai_test")

    # create data version based on current time
    import datetime

    data_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Convert the train and test dataframes back to AzureML datasets
    train_dataset = Data(
        description="RAI train dataset for visit time prediction",
        path="rai_train",
        type=AssetTypes.MLTABLE,
        name="RAI-train",
        version=data_version,
    )
    ml_client.data.create_or_update(train_dataset)
    test_dataset = Data(
        description="RAI test dataset for visit time prediction",
        path="rai_test",
        type=AssetTypes.MLTABLE,
        name="RAI-test",
        version=data_version,
    )
    ml_client.data.create_or_update(test_dataset)

    # Retrieve the latest version of the datasets
    train_dataset = ml_client.data.get("RAI-train", version=data_version)
    test_dataset = ml_client.data.get("RAI-test", version=data_version)

    return train_dataset, test_dataset


def main():

    credential = DefaultAzureCredential()
    try:
        ml_client = MLClient.from_config(credential, path="config.json")

    except Exception as ex:
        print("HERE IN THE EXCEPTION BLOCK")
        print(ex)

    try:
        print(ml_client.compute.get(compute_name))
    except:
        print("No compute found")

    print(os.getcwd())
    print("current", os.listdir())

    # Get handle to azureml registry for the RAI built in components
    # get subscription_id from config.json
    with open("config.json") as f:
        config = json.load(f)
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    workspace_name = config["workspace_name"]
    global ml_client_registry
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        registry_name="azureml",
    )

    train_dataset, test_dataset = create_rai_datasets(ml_client=ml_client)
    # Get the latest version of the model
    versions = [int(m._version) for m in ml_client.models.list(name=model_name)]
    versions.sort(reverse=True)
    global model_id
    model_id = f"{model_name}:{versions[0]}"
    print(f"Using model {model_id}")

    # Pipeline to construct the RAI Insights
    insights_pipeline_job = rai_regression_pipeline(
        target_column_name="VISIT_TIME",
        train_data=train_dataset,
        test_data=test_dataset,
        score_card_config_path=rai_scorecard_config_path,
    )

    insights_job = submit_and_wait(ml_client, insights_pipeline_job)

    # The dashboard should appear in the AzureML portal in the registered model view. The following cell computes the expected URI:

    expected_uri = f"https://ml.azure.com/model/{model_id}/model_analysis?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}"

    print(f"Please visit {expected_uri} to see your analysis")


if __name__ == "__main__":
    main()
