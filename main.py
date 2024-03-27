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
        #data_test = [[0.0, 0.0, 1.0, 0.0, 135000.0, 568800.0, 20560.5, 450000.0, 0.01885, 52.71506849315068, -2329.0, -5170.0, -812.0, 9.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7526144906031748, 0.7896543511176771, 0.1595195404777181, 0.066, 0.059, 0.9732, 0.7552, 0.0211, 0.0, 0.1379, 0.125, 0.2083, 0.0481, 0.0756, 0.0505, 0.0, 0.0036, 0.0672, 0.0612, 0.9732, 0.7648, 0.019, 0.0, 0.1379, 0.125, 0.2083, 0.0458, 0.0771, 0.0526, 0.0, 0.0011, 0.0666, 0.059, 0.9732, 0.7585, 0.0208, 0.0, 0.1379, 0.125, 0.2083, 0.0487, 0.0761, 0.0514, 0.0, 0.0031, 0.0392, 0.0, 0.0, 0.0, 0.0, -1740.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]
        #data = {"data_point": data_test}
        #prediction = sklearn_pyfunc.predict_proba(data_test).max()
            
        model = mlflow.lightgbm.load_model(model_uri="LightGBM")
        
        prediction = model.predict_proba([data.data_point]).max()
        # Get feature importances
        importances = model.feature_importances_
        print(importances)

        return {
            'prediction': prediction,
            'probability': 0.8,
            'importance': importances.tolist()
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