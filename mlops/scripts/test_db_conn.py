import os
import logging
import json
import numpy
import joblib
import pyodbc
import sys

connection_string='Driver={ODBC Driver 17 for SQL Server};Server=tcp:vt-ml-srvr.database.windows.net,1433;Database=vt-ml-db;Uid=vt-sql-admin-login;Pwd=College1//;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;'  

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
               FROM providers WHERE PROVIDERID IN ('000fa702-904f-47dd-b365-f67494102055')"
import pyodbc
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
# cursor.execute('SELECT TOP 1 * FROM providers')
cursor.execute(query)
row = cursor.fetchall()
print(row)
logging.warning("row: %s", row)
conn.close()