import os
import logging
import json
import numpy
import joblib
import pyodbc

connection_string='Driver={ODBC Driver 17 for SQL Server};Server=tcp:vt-ml-srvr.database.windows.net,1433;Database=vt-ml-db;Uid=vt-sql-admin-login;Pwd=College1//;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'  
logging.basicConfig(level=logging.INFO)
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
    
    logging.info("raw_data: ", raw_data)
    data = json.loads(raw_data)["input_data"]    
    logging.info("data: ", data)
    provider_ids = [item["PROVIDERID"] for item in data]
    logging.info("provider_ids: ", provider_ids)
    patient_ids = [item["PATIENTID"] for item in data]
    logging.info("patient_ids: ", patient_ids)
    service_days = [item["SERVICE_DAY"] for item in data]
    logging.info("service_days: ", service_days)
    appt_lats = [item["APPT_LAT"] for item in data]
    logging.info("appt_lats: ", appt_lats)
    appt_lngs = [item["APPT_LNG"] for item in data]
    logging.info("appt_lngs: ", appt_lngs)

    # Convert lists to string format for SQL query
    provider_ids_str = ','.join(map(str, provider_ids))
    logging.info("provider_ids_str: ", provider_ids_str)
    patient_ids_str = ','.join(map(str, patient_ids))
    logging.info("patient_ids_str: ", patient_ids_str)
    # in future, you can add other fields such as evaluation dy of the week.
    # Query provider database

    
    query = f"SELECT \
                    PROVIDERID, \
                    PROVIDERSTATE, \
                    PROVIDERAGE, \
                    HIRINGDATE, \
                    TENURE, \
                    DEGREE, \
                    EMPLOYEETYPENAME, \
                    VISIT_TIME_MEAN, \
                    VISIT_COUNT \
               FROM providers WHERE PROVIDERID IN ({provider_ids_str})"

    logging.info("Provider Query:", query)

    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        provider_data = {row[0]: row for row in cursor.fetchall()}

    # Query patient database
    patient_query = f"SELECT \
                    PATIENTID, \
                    STATE, \
                    CLIENT, \
                    LOB, \
                    GENDERID, \
                    DATEOFBIRTH \
                FROM patients WHERE PATIENTID IN ({patient_ids_str})"

    logging.info("Patient Query:", patient_query)

    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()
        cursor.execute(patient_query)
        patient_data = {row[0]: row for row in cursor.fetchall()}

    # concatenate the values such that the order of columns is:
    #   PROVIDERSTATE, 
    #   PROVIDERAGE, 
    #   HIRINGDATE, 
    #   TENURE, 
    #   DEGREE, 
    #   EMPLOYEETYPENAME, 
    #   VISIT_TIME_MEAN, 
    #   VISIT_COUNT, 
    #   STATE,
    #   CLIENT,
    #   LOB,
    #   GENDERID,
    #   DATEOFBIRTH, 
    #   SERVICE_DAY, 
        #   APPT_LAT, 
        #   APPT_LNG, 
    # this order should be the same as the order of columns in the training data
    input_data = []
    for item in data:
        provider = provider_data[item["PROVIDERID"]]
        patient = patient_data[item["PATIENTID"]]


        input_data.append(numpy.concatenate((provider, patient, service_days, appt_lats, appt_lngs)))
    logging.info("input_data: ", input_data)
    # Predict
    input_data = numpy.array(input_data)
    logging.info("input_data numpy: ", input_data)
    results = model.predict(input_data)
    logging.info("Request processed")
    return results.tolist()