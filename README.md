# credit_scoring_with_MLOPS

We will create a model to predict credit scores and deploy that model as a WebAPP. We use Github Actions to facilitate DevOps.

Tech Stack:
**VS Code** — as the IDE of choice.  
**pipenv** — to handle package dependencies, create virtual environments, and load environment variables.
**FastAPI** — Python API development framework.
	**Uvicorn** — ASGI server for FastAPI app.
**Docker Desktop** — build and run Docker container images on our local machine. (MacOS 11)
    Containers are an isolated environment to run any code
**Azure Container Registry** — repository for storing our container image in Azure cloud.
**Azure App Service** — PaaS service to host our FastAPI app server.
**Github Actions** — automate continuous deployment workflow of FastAPI app.
**Streamlit** - Dashboard