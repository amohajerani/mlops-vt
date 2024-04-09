"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

protected_groups = [
    {
        "feature": "GENDERID",
        "value": 1,
        "type": "categorical",
        "decision_threshold": 1.5,
        "decision_metric": "rmse",
    },
    {
        "feature": "PROVIDERAGE",
        "value": 50,
        "type": "numerical",
        "decision_threshold": 1.5,
        "decision_metric": "rmse",
    },
]
TARGET_COL = "VISIT_TIME"
# Define your categorical and numerical columns
categorical_features = [
    "STATE",
    "CLIENT",
    "LOB",
    "EMPLOYEETYPENAME",
    "PROVIDERSTATE",
    "DEGREE",
]
numerical_features = [
    "VISIT_TIME_MEAN",
    "GENDERID",
    "PROVIDERAGE",
    "TENURE",
    "VISIT_COUNT",
    "APPT_LNG",
    "APPT_LAT",
]


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")

    args = parser.parse_args()

    return args


def main(args):
    """Read trained model and test dataset, evaluate model and save result"""

    # Load the test data
    test_data = pd.read_parquet(Path(args.test_data))

    # Reorder columns
    column_order = [
        "PROVIDERSTATE",
        "PROVIDERAGE",
        "TENURE",
        "DEGREE",
        "EMPLOYEETYPENAME",
        "VISIT_TIME_MEAN",
        "VISIT_COUNT",
        "STATE",
        "CLIENT",
        "LOB",
        "GENDERID",
        "APPT_LAT",
        "APPT_LNG",
    ] + [TARGET_COL]
    test_data = test_data[column_order]

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data.drop(columns=TARGET_COL)

    # Load the model from input port
    model = mlflow.sklearn.load_model(args.model_input)

    # ---------------- Model Evaluation ---------------- #
    yhat_test, score = model_evaluation(X_test, y_test, model, args.evaluation_output)

    # ---------------- Bias Testing ---------------- #

    biased_flag = bias_testing(
        protected_groups, X_test, y_test, yhat_test, args.evaluation_output
    )

    # ----------------- Model Promotion ---------------- #
    predictions, deploy_flag = model_promotion(
        args.model_name,
        args.evaluation_output,
        X_test,
        y_test,
        yhat_test,
        score,
        biased_flag,
    )


def model_evaluation(X_test, y_test, model, evaluation_output):

    # Get predictions to y_test (y_test)
    yhat_test = model.predict(X_test)

    # Save the output data with feature columns, predicted cost, and actual cost in csv file
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((Path(evaluation_output) / "predictions.csv"))

    # Evaluate Model performance with the test set
    r2 = r2_score(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, yhat_test)

    # Print score report to a text file
    (Path(evaluation_output) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}"
    )
    with open((Path(evaluation_output) / "score.txt"), "a") as outfile:
        outfile.write("Model evaluation results on the holdout set: \n")
        outfile.write(f"Mean squared error: {mse:.2f} \n")
        outfile.write(f"Root mean squared error: {rmse:.2f} \n")
        outfile.write(f"Mean absolute error: {mae:.2f} \n")
        outfile.write(f"Coefficient of determination: {r2:.2f} \n")

    mlflow.log_metric("test r2", r2)
    mlflow.log_metric("test mse", mse)
    mlflow.log_metric("test rmse", rmse)
    mlflow.log_metric("test mae", mae)

    # Visualize results
    plt.scatter(y_test, yhat_test, color="black")
    plt.plot(y_test, y_test, color="blue", linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.title("Comparing Model Predictions to Real values - Test Data")
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")

    return yhat_test, r2


def bias_testing(protected_groups, X, y, yhat, evaluation_output):
    """
    Measure the difference in the model performance between different sub-populations.
    This implementation applies to the scenarios where we expect the model prediction to be impacted
    by the protected features (.e.g liver cancer is more likely on men than women).
    We would need a different function for bias testing if we don't expect the protected attribute to impact predictions
    .See the wiki page for context.

    The function calculates model performance on each group (e.g. men vs women), and compares the difference in model performance.

    args:
    protected_groups: list of dict of protected groups: [{'feature': 'age', 'value':50, 'type': 'numerical, decision_threshold': 0.01, 'decision_metric': 'rmse'},]
     - feature: the protected feature to group by
     - value: the value of the protected feature to split the data by
     - type: the type of the protected feature (categorical or numerical)
     - decision_threshold: the threshold of difference between groups to decide if the model is biased
     - decision_metric: the metric to use to decide if the model is biased (e.g. rmse, r2, mae)
    X: dataframe of features
    y: array of target
    yhat: array of predictions
    evaluation_output: path to the evaluation output directory

    returns:
    string: bias testing results

    """
    bias_results = {}
    for group in protected_groups:
        if group["type"] == "categorical":
            mask1 = X[group["feature"]] == group["value"]
            mask2 = X[group["feature"]] != group["value"]
        elif group["type"] == "numerical":
            mask1 = X[group["feature"]] < group["value"]
            mask2 = X[group["feature"]] >= group["value"]
        else:
            raise ValueError("Invalid group type")

        y_group1 = y[mask1]
        yhat_group1 = yhat[mask1]
        y_group2 = y[mask2]
        yhat_group2 = yhat[mask2]

        # Calculate counts and averages
        count_group1 = len(y_group1)
        prediction_avg_group1 = np.mean(yhat_group1)
        target_avg_group1 = np.mean(y_group1)

        count_group2 = len(y_group2)
        prediction_avg_group2 = np.mean(yhat_group2)
        target_avg_group2 = np.mean(y_group2)

        r2_group1 = r2_score(y_group1, yhat_group1)
        mse_group1 = mean_squared_error(y_group1, yhat_group1)
        rmse_group1 = np.sqrt(mse_group1)
        mae_group1 = mean_absolute_error(y_group1, yhat_group1)

        r2_group2 = r2_score(y_group2, yhat_group2)
        mse_group2 = mean_squared_error(y_group2, yhat_group2)
        rmse_group2 = np.sqrt(mse_group2)
        mae_group2 = mean_absolute_error(y_group2, yhat_group2)

        difference_r2 = r2_group2 - r2_group1
        difference_mse = mse_group2 - mse_group1
        difference_rmse = rmse_group2 - rmse_group1
        difference_mae = mae_group2 - mae_group1

        # Decide if the model is biased
        if group.get("decision_metric") == "r2":
            decision = difference_r2 > group.get("decision_threshold")
            difference = difference_r2
        elif group.get("decision_metric") == "mse":
            decision = difference_mse > group.get("decision_threshold")
            difference = difference_mse
        elif group.get("decision_metric") == "rmse":
            decision = difference_rmse > group.get("decision_threshold")
            difference = difference_rmse
        elif group.get("decision_metric") == "mae":
            decision = difference_mae > group.get("decision_threshold")
            difference = difference_mae
        else:
            raise ValueError("Invalid decision metric")
            difference = None

        bias_results[f"{group['feature']}_{group['value']}"] = {
            "count_group1": count_group1,
            "prediction_avg_group1": prediction_avg_group1,
            "target_avg_group1": target_avg_group1,
            "r2_group1": r2_group1,
            "mse_group1": mse_group1,
            "rmse_group1": rmse_group1,
            "mae_group1": mae_group1,
            "count_group2": count_group2,
            "prediction_avg_group2": prediction_avg_group2,
            "target_avg_group2": target_avg_group2,
            "r2_group2": r2_group2,
            "mse_group2": mse_group2,
            "rmse_group2": rmse_group2,
            "mae_group2": mae_group2,
            "difference_r2": difference_r2,
            "difference_mse": difference_mse,
            "difference_rmse": difference_rmse,
            "difference_mae": difference_mae,
            "decision_threshold": group.get("decision_threshold"),
            "decision_metric": group.get("decision_metric"),
            "biased": decision,
        }
        mlflow.log_metric(f"{group['feature']}_{group['value']}_biased", int(decision))
        mlflow.log_metric(
            f"{group['feature']}_{group['value']}_{group.get('decision_metric')}_difference",
            difference,
        )
        mlflow.log_metric(
            f"{group['feature']}_{group['value']}_decision_threshold",
            group.get("decision_threshold"),
        )

    biased = any([results["biased"] for results in bias_results.values()])
    with open((Path(evaluation_output) / "bias_results.txt"), "w") as outfile:
        for group, results in bias_results.items():
            outfile.write(f"{group}:\n")
            outfile.write(f"  Group 1:\n")
            outfile.write(f"    Count: {results['count_group1']}\n")
            outfile.write(
                f"    Prediction Average: {results['prediction_avg_group1']:.2f}\n"
            )
            outfile.write(f"    Target Average: {results['target_avg_group1']:.2f}\n")
            outfile.write(f"    r2: {results['r2_group1']:.2f}\n")
            outfile.write(f"    mse: {results['mse_group1']:.2f}\n")
            outfile.write(f"    rmse: {results['rmse_group1']:.2f}\n")
            outfile.write(f"    mae: {results['mae_group1']:.2f}\n")
            outfile.write(f"  Group 2:\n")
            outfile.write(f"    Count: {results['count_group2']}\n")
            outfile.write(
                f"    Prediction Average: {results['prediction_avg_group2']:.2f}\n"
            )
            outfile.write(f"    Target Average: {results['target_avg_group2']:.2f}\n")
            outfile.write(f"    r2: {results['r2_group2']:.2f}\n")
            outfile.write(f"    mse: {results['mse_group2']:.2f}\n")
            outfile.write(f"    rmse: {results['rmse_group2']:.2f}\n")
            outfile.write(f"    mae: {results['mae_group2']:.2f}\n")
            outfile.write(f"  Difference:\n")
            outfile.write(f"    r2: {results['difference_r2']:.2f}\n")
            outfile.write(f"    mse: {results['difference_mse']:.2f}\n")
            outfile.write(f"    rmse: {results['difference_rmse']:.2f}\n")
            outfile.write(f"    mae: {results['difference_mae']:.2f}\n")
            outfile.write(f"  Decision threshold: {results['decision_threshold']}\n")
            outfile.write(f"  Decision metric: {results['decision_metric']}\n")
            outfile.write(f"  Biased: {results['biased']}\n")
        outfile.write(f"Overall bias test result:\nBiased: {biased}\n")
    mlflow.log_metric("biased", int(biased))
    mlflow.log_artifact((Path(evaluation_output) / "bias_results.txt"))

    return biased


def model_promotion(
    model_name, evaluation_output, X_test, y_test, yhat_test, score, biased
):
    """
    # Compare the current model with the latest version of the model in the registry.
    # TODO: uncomment this piece

    scores = {}
    predictions = {}

    client = MlflowClient()
    # Get the latest version of the model
    model_run = client.search_model_versions(f"name='{model_name}'")[0]
    model_version = model_run.version

    mdl = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}")
    predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
    scores[f"{model_name}:{model_version}"] = r2_score(
        y_test, predictions[f"{model_name}:{model_version}"])

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    with open((Path(evaluation_output) / "deploy_flag"), 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # add current model score and predictions
    scores["current model"] = score
    predictions["currrent model"] = yhat_test

    perf_comparison_plot = pd.DataFrame(
        scores, index=["r2 score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(evaluation_output) / "perf_comparison.png")


    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    return predictions, deploy_flag
    """
    poor_performance = 0  # 0 means there are no issues to be flagged
    if poor_performance or biased:
        deploy_flag = 0
    else:
        deploy_flag = 1

    logger.info(
        f"Deploy flag: {deploy_flag}, Poor performance: {poor_performance}, Biased: {biased}"
    )

    with open((Path(evaluation_output) / "deploy_flag"), "w") as outfile:
        outfile.write(f"{int(deploy_flag)}")
    return None, deploy_flag


if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_input}",
        f"Test data path: {args.test_data}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
