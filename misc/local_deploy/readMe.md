Test and debug of deployment can be time consuming. This is because every change has to go through a CI/CD pipeline run, which can take a good 10-20 minutes. Futher, debugging is complicated by lack of complete logs.

To iterate quickly, and have a better visibility to logs, we can deploy the ML model locally. 

Follow these steps:
- Build the docker image using build_img.sh. This image provides a workaround for ssl certificate error that you'd otherwise may experience.
- store the image in azure container repository: 
    Login to the ACR: az acr login --name myRegistry
    Tag your Docker image with the login server name of the ACR (replace 'myRegistry' with your own ACR name. For my personal account it is crvtpoc9dev): docker tag local_deploy_img myRegistry.azurecr.io/local_deploy_img:v1
    Push the Docker image to the ACR: docker push myRegistry.azurecr.io/local_deploy_img:v1
- ensure the reference to the docker image in your python script env definition is correct.
- In the scoring script, revise the path to the model.pkl
- Run the python script. 

to see the logs, find the container name and then run docker logs on that container. alternatively use the get_logs command