import os
import logging
import json
import numpy
import joblib
import pyodbc

connection_string='Driver={ODBC Driver 18 for SQL Server};Server=tcp:vt-ml-srvr.database.windows.net,1433;Database=vt-ml-db;Uid=vt-sql-admin-login;Pwd=College1//;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'  

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
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["input_data"]    
    provider_ids = [item["providerId"] for item in data]
    patient_ids = [item["patientId"] for item in data]
    # Convert lists to string format for SQL query
    provider_ids_str = ','.join(map(str, provider_ids))
    patient_ids_str = ','.join(map(str, patient_ids))
    # in future, you can add other fields such as evaluation dy of the week.
    # Query provider database
    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, Age FROM Providers WHERE id IN ({provider_ids_str})")
        provider_data = {row[0]: row for row in cursor.fetchall()}

    # Query patient database
    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, Age FROM Patients WHERE id IN ({patient_ids_str})")
        patient_data = {row[0]: row for row in cursor.fetchall()}

    # Concatenate provider and patient data and predict
    input_data = []
    for item in data:
        provider = provider_data[item["providerId"]]
        patient = patient_data[item["patientId"]]
        input_data.append(numpy.concatenate((provider, patient)))

    # Predict
    input_data = numpy.array(input_data)
    results = model.predict(input_data)
    logging.info("Request processed")
    return results.tolist()