# End-to-end machine learning using FastAPI, Streamlit, Docker, Microsoft AZURE

Objective of the ML model is to predict a credit score.

We will create a model to predict credit scores and deploy that model as a WebAPP. We use Github Actions to facilitate DevOps.

Tech Stack:  
**VS Code** — as the IDE of choice.  
**pipenv** — to handle package dependencies, create virtual environments, and load environment variables.  
**FastAPI** — Python API development framework for ML deployment 
	**Uvicorn** — ASGI server for FastAPI app.  
**Docker Desktop** — build and run Docker container images on our local machine. (MacOS 11)  
    Containers are an isolated environment to run any code  
**Azure Container Registry** — repository for storing our container image in Azure cloud.  
**Azure App Service** — PaaS service to host our FastAPI app server.  
**Github Actions** — automate continuous deployment workflow of model serving through FastAPI and dashboarding through Streamlit app.  
**Streamlit** - Dashboard  

We will cover in the readme the below concepts:

1) How to create dockerfile for ML API deployment using FastAPI?
2) How to run different docker commands to build, run and debug and ?
3) How to push docker image to Github using Github Actions
4) How to test ML API endpoint which is exposed by the running ML API docker container?

The data  
credit  


The backend


The frontend  