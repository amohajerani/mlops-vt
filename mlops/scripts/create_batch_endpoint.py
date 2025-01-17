import argparse

from azure.ai.ml.entities import BatchEndpoint

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Create batch endpoint")
    parser.add_argument("--endpoint_name", type=str, help="Name of batch endpoint")
    parser.add_argument("--description", type=str, help="Description of the batch endpoint")
    parser.add_argument("--auth_mode", type=str, help="auth_mode")

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    
    credential = DefaultAzureCredential()
    try:
        ml_client = MLClient.from_config(credential, path='config.json')

    except Exception as ex:
        print("HERE IN THE EXCEPTION BLOCK")
        print(ex)

    # create batch endpoint
    batch_endpoint = BatchEndpoint(
        name=args.endpoint_name, 
        description=args.description,
        auth_mode=args.auth_mode
    )
    
    endpoint_job = ml_client.batch_endpoints.begin_create_or_update(batch_endpoint)
    endpoint_job.wait()


if __name__ == "__main__":
    main()
