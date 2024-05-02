import argparse
from operator import attrgetter
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def parse_args():
    parser = argparse.ArgumentParser(description="Delete old deployments")
    parser.add_argument("--endpoint_name", type=str, help="online endpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    credential = DefaultAzureCredential()
    try:
        ml_client = MLClient.from_config(credential, path="config.json")

    except Exception as ex:
        print("HERE IN THE EXCEPTION BLOCK")
        print(ex)

    # Get the deployments
    deployments = ml_client.online_deployments.list(endpoint_name=args.endpoint_name)
    deployments_list = list(deployments)
    deployments_list.sort(key=attrgetter("name"))

    # Delete all but the last two deployments
    for deployment in deployments_list[:-2]:
        print(f"Deleting deployment: {deployment.name}")
        ml_client.online_deployments.begin_delete(
            name=deployment.name, endpoint_name=args.endpoint_name
        ).result()


if __name__ == "__main__":
    main()
