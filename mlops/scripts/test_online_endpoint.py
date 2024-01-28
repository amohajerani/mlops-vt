import argparse

from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import ManagedOnlineDeployment

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Test online endpoint")
    parser.add_argument("--endpoint_name", type=str, help="Name of the online endpoint")
    parser.add_argument("--request_file", type=str, help="Path of the request json file")
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

    # invoke and test endpoint
    response = ml_client.online_endpoints.invoke(
        endpoint_name=args.endpoint_name,
        request_file=args.request_file
    )
    # Print the response
    print("Response:", response)


    # Check the response
    if 'error' in response:
        print("Test failed, received error:", response['error'])
    else:
        print("Test passed, received expected response")


if __name__ == "__main__":
    main()
