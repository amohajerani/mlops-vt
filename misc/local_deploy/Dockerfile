# Use the specified image as the base
# replace this base image with whatever you are using in mlops/scripts/register_environment.py
FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04

# This is a workaround for the SSL issue that occurs when trying to install packages from conda

RUN conda config --set ssl_verify false