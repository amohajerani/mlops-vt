# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# enter details of your AML workspace
subscription_id = "ea6d544d-2425-4667-8d38-51f050d9d69e"
resource_group = "rg-vtpoc-9dev"
workspace = "mlw-vtpoc-9dev"

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Define an endpoint name
endpoint_name = "my-endpoint"

# Example way to define a random name
import datetime

endpoint_name = "endpt-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="this is a sample online endpoint",
    auth_mode="key",
    tags={"foo": "bar"},
)

model = Model(path="vt-model/model.pkl")
env = Environment(
    conda_file="../data-science/environment/train-conda.yml",
    image="crvtpoc9dev.azurecr.io/local_deploy_img:v1",
)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="../mlops/scripts", scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)

ml_client.online_deployments.begin_create_or_update(
    deployment=blue_deployment, local=True
)

ml_client.online_endpoints.get(name=endpoint_name, local=True)

ml_client.online_deployments.get_logs(
    name="blue", endpoint_name=endpoint_name, local=True, lines=50
)

ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    request_file="../data/test-request.json",
    local=True,
)
