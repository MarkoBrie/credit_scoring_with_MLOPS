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
        #prediction = sklearn_pyfunc.predict_proba(data_test).max()
            
        model = mlflow.lightgbm.load_model(model_uri="LightGBM")
        
        prediction = model.predict_proba([data.data_point]).max()
        # Get feature importances
        importances = model.feature_importances_

        return {
            'prediction': prediction,
            'probability': 0.8
        }
    except Exception as e:
        error_msg = f"An error occurred during prediction: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
def index():
    return {"data": "Application ran successfully - FastAPI ML endpoint deployed with Github Actions on Microsoft AZURE"     
    }
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)