import os
import json
import numpy
import joblib
import pandas as pd
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("score.py")


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "vt-model/model.pkl")
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


@input_schema(
    param_name="input_data",
    param_type=PandasParameterType(
        pd.DataFrame(
            {
                "GENDERID": [2.0, 1.0],
                "EMPLOYEETYPENAME": ["1099 Contractor", "1099 Contractor"],
                "PROVIDERSTATE": ["AL", "TX"],
                "PROVIDERAGE": [42.0, 63.0],
                "TENURE": [267.0, 198.0],
                "DEGREE": ["NP", "DO"],
                "VISIT_TIME_MEAN": [29, 42],
                "VISIT_COUNT": [319, 489],
            }
        )
    ),
)
@output_schema(output_type=StandardPythonParameterType([45.00, 37.7]))
def run(input_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back.

    The PROVIDERSTATE is supplied by client but not used in the model.
    """

    column_names = [
        "PROVIDERAGE",
        "TENURE",
        "DEGREE",
        "EMPLOYEETYPENAME",
        "VISIT_TIME_MEAN",
        "VISIT_COUNT",
        "GENDERID",
    ]

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame(input_data)
    # filter the columns to the column_names, and order them in the same order in the column_names
    try:
        input_data = input_data[column_names]
    except KeyError as e:
        logger.error(f"Error: {e}")
        return [-1] * len(input_data)

    # Predict
    results = model.predict(input_data)
    return results.tolist()
