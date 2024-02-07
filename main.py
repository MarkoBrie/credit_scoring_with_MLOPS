import uvicorn
from fastapi import FastAPI
import os
#import streamlit as st

# load environment variables
port = os.environ["PORT"]

# initialize FastAPI
app = FastAPI(title="Automatic Credit Scoring",
              description='''Obtain a credit score (0,1) for ClientID.
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",)

@app.get("/")
def index():
    return {"data": "Application ran successfully - FastAPI release v4.2 with Github Actions no staging",
            'prediction': 1,
            'probability': 0.9
            
            }
    #return {st.title("Hello World")}
  
  
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)