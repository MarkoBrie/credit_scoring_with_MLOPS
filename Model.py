!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:08:50 2024

@author: markobriesemann
"""

# 1. Library imports
import pandas as pd 
import joblib
import mlflow

sklearn_pyfunc = mlflow.lightgbm.load_model(model_uri="mlflow_model_LightGBM")


def predict_species():
        model_fname_ = 'model.pkl'
        model = joblib.load(self.model_fname_)
        data_in = 
        prediction = model.predict(data_in)
        probability = model.predict_proba(data_in).max()
        return prediction[0], probability

