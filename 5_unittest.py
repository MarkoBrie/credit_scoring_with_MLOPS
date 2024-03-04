#import unittest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from main import app
import pytest
import os
from unittest.mock import patch

client = TestClient(app)

#class TestConnection(unittest.TestCase):
@patch.dict(os.environ, {'PORT' : '8000'})
def test_response_withData():
    """
    TEST model output with fixture test data 
    """

    # fixture simulation with test data
    data_for_request =  [0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                                                            False]
    # Send the POST request with the data
    response = client.post("/predict", json={"data_point": data_for_request})
    assert response.status_code == 200
    assert response.json() == {"prediction":0.857982822560715,"probability":0.8}
    # Unit tests for response status codes

@patch.dict(os.environ, {'PORT' : '8000'})
def test_client_emptyData():
    """
    TEST "post" with empty data 
    """
    # fixture simulation with test data
    data_for_request =  []
    
    # Send the POST request with the data
    response = client.post("/predict", json={"data_point": data_for_request})
    assert response.status_code == 500
    assert response.json() == {"detail":"An error occurred during prediction: Found array with 0 feature(s) (shape=(1, 0)) while a minimum of 1 is required."}

@patch.dict(os.environ, {'PORT' : '8000'})
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"data": "Application ran successfully - FastAPI ML endpoint deployed with Github Actions on Microsoft AZURE"}

if __name__ == '__main__':
    unittest.main()