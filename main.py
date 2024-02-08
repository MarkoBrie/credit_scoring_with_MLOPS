import uvicorn
from fastapi import FastAPI
#import pickle # pipfile does not lock
import mlflow
import lightgbm
import os
#import streamlit as st
#import cloudpickle
#import pickle

# load environment variables
port = os.environ["PORT"]

# initialize FastAPI
app = FastAPI(title="Automatic Credit Scoring",
              description='''Obtain a credit score (0,1) for ClientID.
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",)

# Path to the .pkl file containing the serialized model
#model_file_path = "model.pkl"

# Load the model from the .pkl file
#with open(model_file_path, 'rb') as f:
 #   model = pickle.load(f)

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_credit_score():

    sklearn_pyfunc = mlflow.lightgbm.load_model(model_uri="LightGBM")
    data = [[0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]

    prediction = sklearn_pyfunc.predict_proba(data).max()
    print(prediction)
    return {
        'prediction': prediction,
        'probability': 0.9
    }

@app.get("/")
def index():
    return {"data": "Application ran successfully - FastAPI release v4.2 with Github Actions no staging: cloudpickle try environment pipenv",
            
    }
    #return {st.title("Hello World")}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)