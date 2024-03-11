# End-to-end machine learning using FastAPI, Streamlit, Docker, GitHub Actions, Microsoft AZURE and EvidentlyAI

The **Objective** of the ML model is to predict a credit score for historical loan application data to predict whether or not an applicant will be able to repay a loan.

We use a pretrained model to predict credit scores and deploy that model as an API using FastAPI and use Streamlit to create a WebAPP. We use Github Actions to facilitate and automatize DevOps and MLOps (CI/CD).

[**Overview**](#overview)
| [**Tech Stack**](#tech-stack)
| [**The data**](#the-data)
| [**The backend with FastAPI**](#the-backedn-with-fastapi)
| [**Docker**](#docker)
| [**Hosting with Microsoft AZURE**](#hosting-with-microsoft-azure)
| [**Github Actions**](#github-actions)

## Tech Stack:  
**VS Code** — as the IDE of choice.  
**pipenv** — to handle package dependencies, create virtual environments, and load environment variables.  
**FastAPI** — Python API development framework for ML endpoint deployment 
	**Uvicorn** — ASGI server for FastAPI app.  
**Docker Desktop** — build and run Docker container images on our local machine. (MacOS 11)  
    Containers are an isolated environment to run any code  
**Azure Container Registry** — repository for storing our container image in Azure cloud.  
**Azure App Service** — PaaS service to host our FastAPI app server.  
**Github Actions** — automate continuous deployment workflow of model serving through FastAPI and dashboarding through Streamlit app.  
**Streamlit** - Dashboard  
**PyTest** - Testing of Web APP functionality through
**EvidentlyAI** - Data Drift detection

## Scope of Readme

We will cover in the readme the below concepts:


1) How to create dockerfile for ML API deployment using FastAPI
2) How to run different docker commands to build, run and debug and 
3) How to push docker image to Github using Github Actions
4) How to test ML API endpoint which is exposed by the running ML API docker container

## The data
The data is provided by [Home Credit](http://www.homecredit.net/about-us.aspx).

## Project setup 

### Create project folder and start Visual Studio Code
Open a terminal 
```
mkdir project_name
cd project_name
code .
```
### Pipenv environment for package dependency management
Install pipenv
```
sudo -H pip install -U pipenv
pipenv install fastapi uvicorn
```

use `pipenv shell` to activate the virtual environment.


## The backend with FastAPI
You find the FastAPI in the file main.py
```
import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import mlflow
import lightgbm
import os
from typing import List
from pydantic import BaseModel # for data validation

# load environment variables
#port = os.environ["PORT"]
# hard coded port to facilitate testing with Github Actions
port = 8000

# initialize FastAPI
app = FastAPI(title="Automatic Credit Scoring",
              description='''Obtain a credit score (0,1) for ClientID.
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",)

# Pydantic model for the input data
class DataPoint(BaseModel):
    data_point: List[float]

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_credit_score(data: DataPoint):
    """ Endpoint for ML model

    Args:
                    list (float): one data point of 239 floats
                    
    Returns:
                float: prediction probability
                int: prediction score
    """
    try:
        print("predict_credit_score function")
        #print(data)
        print([data.data_point])

        if (len([data.data_point]) == 0):
            print("empty data set")
            return {"VALUE ERROR": "Data set is empty"     
        }

        # TEST data  
        #data_test = [[0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]
        #data = {"data_point": data_test}

        sklearn_pyfunc = mlflow.lightgbm.load_model(model_uri="LightGBM")
        
        prediction = sklearn_pyfunc.predict_proba([data.data_point]).max()

        return {
            'prediction': prediction,
            'probability': 0.8
        }
    except Exception as e:
        error_msg = f"An error occurred during prediction: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
def index():
    return {"data": "Application ran successfully - FastAPI release v4.2 with Github Actions no staging: cloudpickle try environment pipenv"     
    }
    #return {st.title("Hello World")}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
```

Run your APP with
```
pipenv run python main.py
```

You can find more details on creating and deploying a FastAPI in this [blog](https://towardsdatascience.com/how-to-build-and-deploy-a-machine-learning-model-with-fastapi-64c505213857)

## Docker
In order to deploy our API endpoint to Azure we need to install (Docker Desktop)[https://www.docker.com/products/docker-desktop].

In the ```Dockerfile``` you need to add the libraries that your APP needs to run.
For example ```RUN pip install uvicorn```.
For our FASTAPI API we use ```Python 3.8-bookworm```.

Here are some useful commands to create a docker image:
```
docker build . -t fastapi-cd:1.0
````
or to run the API from the docker image
```
docker run -p 8000:8000 -t fastapi-cd:1.0
```
You can check the API at ```http://localhost:8000/docs```.



## Hosting with Microsoft AZURE

You find a great tutorial on how to setup your infrastructure [here](https://towardsdatascience.com/deploy-fastapi-on-azure-with-github-actions-32c5ab248ce3)
 
## Github Actions
Using Github Actions we can automatize the Docker build process and the push to our hosting platform. We have created a yml file in the 
`.github/workflows` folder.
```
docker build . -t fastapicd2024.azurecr.io/fastapi-cd:${{ github.sha }}
docker push fastapicd2024.azurecr.io/fastapi-cd:${{ github.sha }}
```
Details on how to setup your Github Actions with Azure you can find in this [blog](https://towardsdatascience.com/deploy-fastapi-on-azure-with-github-actions-32c5ab248ce3)

## The frontend Dashboard with Streamlit

You can run the dashboard locally anytime with this command but be aware that you will not get any prediction as long as the backend is not running. So make sure to start the back end first if you wish to test the dashboard together with the ML endpoint.

```
streamlit run 3_STREAMlit_dashboard.py
`````

The Streamlit APP will be visible in your browser at: http://localhost:8501

### Streamlit and Docker

Make sure ```Docker Desktop``` is running. Go to the streamlit dashboard folder ```3_STREAMlit_dashboard```. Then run the following command in the terminal to build the container image:

```
docker build . -t dashboard-cd:1.0
```
and the next command to test the container image:
```
docker run -p 8501:8501 -t dashboard-cd:1.0
```

In Microsoft AZURE open ```AZURE Container Registry```
Create a new one with the following parameters:

```
Resource group: fastapi-cd
Container group: fastapicd2024
Registry name: fastapi-cd
Login server: fastapicd2024.azurecr.io
```

Open Terminal and run:
```
az login
az acr login --name fastapicd2024
```
then try to build the container that is pushed to AZURE:
```
docker build . -t fastapicd2024.azurecr.io/dashboard-cd:1.1
docker images
docker push fastapicd2024.azurecr.io/dashboard-cd:1.1
```

Now you just need to create a Web APP service instance and connect to your container:
Go to ```AZURE APP Service``` and create a new instance:

In Basics tab 
``` 
Resource group: fastapi-cd
Name: credit-score-dashboard
Publier: Container Docker  
```

And in the Docker tab  
```
Image source: Azure Container Registry
Registry: fastapicd2024
Image: dashboard-cd
Startup command: streamlit run 3_STREAMlit_dashboard.py
```

Now got to to your WEB APP at: 
https://credit-score-dashboard.azurewebsites.net/

## Python code test with Pytest

You find the python code test in the file ```5_unittest.py````

