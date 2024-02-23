#import unittest
import requests
from fastapi.testclient import TestClient
import httpx
from main import app
import pytest


client = TestClient(app)

#class TestConnection(unittest.TestCase):
def test_connection_functionality():
    """
    Test that connection is working and 
    """
    try:
        test_location = "local"
        if (test_location == "local"):
            host = '127.0.0.1'
            port = '8000'
            # endpoint
            url = f'http://{host}:{port}/predict'
        else:
            url = 'https://fastapi-cd-webapp.azurewebsites.net/predict'
        
        data_for_request =  [0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                                                                False]
        # Send the POST request with the data
        response = requests.post(url, json={"data_point": data_for_request})
        assert response.status_code == 200
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_response_withData():
    """
    TEST model output with fixture test data 
    """
    try:
        test_location = "local"
        if (test_location == "local"):
            host = '127.0.0.1'
            port = '8000'
            # endpoint
            url = f'http://{host}:{port}/predict'
        else:
            url = 'https://fastapi-cd-webapp.azurewebsites.net/predict'
        
        # fixture simulation with test data
        data_for_request =  [0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                                                                False]
        # Send the POST request with the data
        response = requests.post(url, json={"data_point": data_for_request})
        assert response.status_code == 200
        assert response.json() == {"prediction":0.857982822560715,"probability":0.8}
        # Unit tests for response status codes
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_response_emptyData():
    """
    TEST "post" with empty data 
    """
    try:
        test_location = "local"
        if (test_location == "local"):
            host = '127.0.0.1'
            port = '8000'
            # endpoint
            url = f'http://{host}:{port}/predict'
        else:
            url = 'https://fastapi-cd-webapp.azurewebsites.net/predict'
        
        # fixture simulation with test data
        data_for_request =  []
        
        # Send the POST request with the data
        response = requests.post(url, json={"data_point": data_for_request})
        assert response.status_code == 500
        assert response.json() == {"detail":"An error occurred during prediction: Found array with 0 feature(s) (shape=(1, 0)) while a minimum of 1 is required."}
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

#if __name__ == '__main__':
#    unittest.main()