import os
import json
import numpy
import joblib
import pyodbc
import pandas as pd
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)

connection_string='Driver={ODBC Driver 17 for SQL Server};Server=tcp:vt-ml-srvr.database.windows.net,1433;Database=vt-ml-db;Uid=vt-sql-admin-login;Pwd=College1//;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;'  

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "vt-model/model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

@input_schema(
    param_name="input_data",
    param_type=PandasParameterType(
        pd.DataFrame(
            {
                "PROVIDERIDS": [
                  "000fa702-904f-47dd-b365-f67494102055",
                  "000fa702-904f-47dd-b365-f67494102055"
                ],
                "PATIENTIDS": [
                  "0021E536-383A-48E2-BC76-EA5C7EAC24F9",
                  "0021E536-383A-48E2-BC76-EA5C7EAC24F9"
                ],
                "APPT_LATS": [
                  37.7749,
                  34.0522
                ],
                "APPT_LNGS": [
                  -102.4194,
                  -118.2437
                ]
              }
        )
    ),
)
@output_schema(output_type=StandardPythonParameterType([45.00]))

def run(input_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    PROVIDERIDS = input_data["PROVIDERIDS"]
    PATIENTIDS = input_data["PATIENTIDS"]
    APPT_LATS = input_data["APPT_LATS"]
    APPT_LNGS = input_data["APPT_LNGS"]
    # Convert lists to string format for SQL query
    provider_ids_str = ','.join(f"'{id}'" for id in PROVIDERIDS)
    patient_ids_str = ','.join(f"'{id}'" for id in PATIENTIDS)
    # in future, you can add other fields such as evaluation dy of the week.
    # Query provider database

    cnt = len(PROVIDERIDS)
    if cnt != len(PATIENTIDS) or cnt != len(APPT_LATS) or cnt != len(APPT_LNGS):
        raise ValueError("Length of PROVIDERIDS, PATIENTIDS, APPT_LATS, and APPT_LNGS must be the same")
    
    query = f"SELECT \
                    PROVIDERID, \
                    PROVIDERSTATE, \
                    PROVIDERAGE, \
                    TENURE, \
                    DEGREE, \
                    EMPLOYEETYPENAME, \
                    VISIT_TIME_MEAN, \
                    VISIT_COUNT \
               FROM providers WHERE PROVIDERID IN ({provider_ids_str})"


    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    cursor.execute(query)
    provider_data = {row[0]: row for row in cursor.fetchall()}

    # Query patient database
    patient_query = f"SELECT \
                    PATIENTID, \
                    STATE, \
                    CLIENT, \
                    LOB, \
                    GENDERID \
                FROM patients WHERE PATIENTID IN ({patient_ids_str})"


    cursor.execute(patient_query)
    patient_data = {row[0]: row for row in cursor.fetchall()}
    conn.close()

    # concatenate the values such that the order of columns is:
    #   PROVIDERSTATE, 
    #   PROVIDERAGE, 
    #   TENURE, 
    #   DEGREE, 
    #   EMPLOYEETYPENAME, 
    #   VISIT_TIME_MEAN, 
    #   VISIT_COUNT, 
    #   STATE,
    #   CLIENT,
    #   LOB,
    #   GENDERID,
    #   APPT_LAT, 
    #   APPT_LNG, 
    # this order should be the same as the order of columns in the training data
    input_data = []
    for i in range(cnt):
        provider_info = provider_data[PROVIDERIDS[i]][1:]
        patient_info = patient_data[PATIENTIDS[i]][1:]
        input_data.append(numpy.concatenate((provider_info, patient_info, [APPT_LATS[i]], [APPT_LNGS[i]])))
    

    column_names = [
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
    "APPT_LNG"
    ]

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame(input_data, columns=column_names)

    # Predict
    results = model.predict(input_data)
    return results.tolist()