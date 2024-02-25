Test and debug of deployment can be time consuming. This is because every change has to go through a CI/CD pipeline run, which can take a good 10-20 minutes.

To iterate quickly, and have a better visibility to logs, we can deploy the ML model locally.
Follow these steps:
- Build the docker image using build_img.sh. This image provides a workaround for ssl certificate error that you'd otherwise may experience.
- Run the python script.
