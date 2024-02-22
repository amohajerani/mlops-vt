"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


TARGET_COL = "VISIT_TIME"
# Define your categorical and numerical columns
categorical_features = ['STATE', 'CLIENT', 'LOB', 'EMPLOYEETYPENAME', 'PROVIDERSTATE', 'DEGREE']
numerical_features = ['PROD_CKD', 'PROD_PAD', 'VISIT_TIME_MEAN', 'PROD_HHRA', 'GENDERID', 'PROD_MHC', 'PROVIDERAGE', 'PROD_DEE', 'TENURE', 'VISIT_COUNT', 'PROD_DSNP', 'PROD_SPIROMETRY', 'PROD_OMW', 'PROD_FOBT', 'PROD_HBA1C', 'APPT_LNG', 'APPT_LAT', 'PROD_MTM']


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")

    args = parser.parse_args()

    return args

def main(args):
    '''Read trained model and test dataset, evaluate model and save result'''

    # Load the test data
    test_data = pd.read_parquet(Path(args.test_data))

    # Reorder columns
    column_order = [        
        'PROVIDERSTATE', 
        'PROVIDERAGE', 
        'HIRINGDATE', 
        'TENURE', 
        'DEGREE', 
        'EMPLOYEETYPENAME', 
        'VISIT_TIME_MEAN', 
        'VISIT_COUNT', 
        'STATE', 
        'SERVICE_DAY', 
        'APPT_LAT', 
        'APPT_LNG', 
        'CLIENT', 
        'LOB', 
        'GENDERID', 
        'DATEOFBIRTH',
        'SERVICE_DAY',
        'APPT_LAT',
        'APPT_LNG'] + [TARGET_COL]
    test_data = test_data[column_order]

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data.drop(columns=TARGET_COL)

    # Load the model from input port
    model =  mlflow.sklearn.load_model(args.model_input) 

    # ---------------- Model Evaluation ---------------- #
    yhat_test, score = model_evaluation(X_test, y_test, model, args.evaluation_output)

    # ----------------- Model Promotion ---------------- #
    predictions, deploy_flag = model_promotion(args.model_name, args.evaluation_output, X_test, y_test, yhat_test, score)



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
        outfile.write("Mean squared error: {mse.2f} \n")
        outfile.write("Root mean squared error: {rmse.2f} \n")
        outfile.write("Mean absolute error: {mae.2f} \n")
        outfile.write("Coefficient of determination: {r2.2f} \n")

    mlflow.log_metric("test r2", r2)
    mlflow.log_metric("test mse", mse)
    mlflow.log_metric("test rmse", rmse)
    mlflow.log_metric("test mae", mae)

    # Visualize results
    plt.scatter(y_test, yhat_test,  color='black')
    plt.plot(y_test, y_test, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.title("Comparing Model Predictions to Real values - Test Data")
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")

    return yhat_test, r2

def model_promotion(model_name, evaluation_output, X_test, y_test, yhat_test, score):
    '''
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
    '''
    deploy_flag = 1
    with open((Path(evaluation_output) / "deploy_flag"), 'w') as outfile:
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
