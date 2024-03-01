# End-to-end machine learning using FastAPI, Streamlit, Docker, GitHub Actions, Microsoft AZURE and EvidentlyAI

The **Objective** of the ML model is to predict a credit score fro historical loan application data to predict whether or not an applicant will be able to repay a loan.

We will create a model to predict credit scores and deploy that model as an API using FastAPI and use Streamlit to create a WebAPP. We use Github Actions to facilitate and automatize DevOps and MLOps (CI/CD).

## Tech Stack:  
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
**PyTest** - Testing of Web APP functionality through
**EvidentlyAI** - Data Drift detection

# Outline of Readme

We will cover in the readme the below concepts:


1) How to create dockerfile for ML API deployment using FastAPI?
2) How to run different docker commands to build, run and debug and 
3) How to push docker image to Github using Github Actions
4) How to test ML API endpoint which is exposed by the running ML API docker container?

## The data
The data is provided by [Home Credit](http://www.homecredit.net/about-us.aspx).

## Project setup 

### Create project folder and start Visual Studio Code
Open a terminal 
'''
mkdir project_name
cd project_name
code .
'''
### Pipenv environment
Install pipenv
'''
sudo -H pip install -U pipenv
pipenv install fastapi uvicorn
'''

## Hosting with Microsoft AZURE


## The backend with Streamlit


## The frontend with FastAPI

## Docker