import argparse
import snowflake.connector
import pandas as pd
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import time


# Set up logging
logging.basicConfig(level=logging.INFO)

# Create the parser
parser = argparse.ArgumentParser(description='Data import script')

# Add the arguments
parser.add_argument('workspace_name', type=str, help='Azure ML workspace name')
parser.add_argument('snowflake_account', type=str, help='Snowflake account')
parser.add_argument('snowflake_user', type=str, help='Snowflake user')
parser.add_argument('snowflake_password', type=str, help='Snowflake password')
parser.add_argument('subscription_id', type=str, help='Azure subscription ID')
parser.add_argument('resource_group', type=str, help='Azure resource group')

# Parse the arguments
args = parser.parse_args()

logging.info('Arguments parsed')
# Authenticate to Azure ML
ml_client = MLClient(
    DefaultAzureCredential(), args.subscription_id, args.resource_group, args.workspace_name
)

# verify we are connected
ws = ml_client.workspaces.get(args.workspace_name)
logging.info('Verifying client connection to the workspace :',ws.location,":", ws.resource_group)


# Snowflake connection parameters
snowflake_account = args.snowflake_account
snowflake_user = args.snowflake_user
snowflake_password = args.snowflake_password

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account
)

logging.info('Connected to Snowflake')

# Execute a query
cur = conn.cursor()
cur.execute("SELECT * FROM vt.public.train")
rows = cur.fetchall()

logging.info('Fetched data from Snowflake')

# Convert to pandas DataFrame
df = pd.DataFrame(rows, columns=[x[0] for x in cur.description])

logging.info('Converted data to pandas DataFrame')

# store the file as a csv file
df.to_csv('train.csv', index=False)


# instantiate a data object
version = "v" + time.strftime("%Y.%m.%d.%H%M%S", time.gmtime())
dataset = Data(
    name='train',
    description='train dataset',
    version=version,
    path='train.csv',
    type=AssetTypes.URI_FILE
)

# create data asset on Azure ML
ml_client.data.create_or_update(dataset)

# verify the dataset is registered
datasets = ml_client.data.list()
for d in datasets:
    if d.name == 'train':
        logging.info('Dataset registered in Azure ML workspace')
        break
