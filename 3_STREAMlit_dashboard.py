import pandas as pd
import streamlit as st
import requests

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    st.write(data)
    data_json = data

    st.write(data_json)
    
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()
#load id_test file
#

def main():
    MLFLOW_URI = 'https://fastapi-cd-webapp.azurewebsites.net/predict'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['MLflow', 'Option 2', 'Option 3'])

    st.title('Median House Price Prediction')

    selected_radio = st.radio('Select an option', ['Option 1', 'Option 2', 'Option 3'])

    # List of IDs
    id_list = ['100002', '100003', '100004', '100005', '100006', '100007', '100008', '100009']

    selected_id = st.selectbox('Search and select an ID', options=id_list, index=0, format_func=lambda x: x if x else 'Search...')
    #get selected_id index in ids_test and use the index to get the data

    revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
                                 min_value=0., value=3.87, step=1.)

    predict_btn = st.button('Prédire')
    if predict_btn:
        data = pd.DataFrame([[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
                 taille_pop, occupation_moy, latitude, longitude]])#.to_json(orient='records')
        
        data = {"dataframe_records": [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
                 taille_pop, occupation_moy, latitude, longitude]]}
        
        data = { "inputs":[[0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]}

        
        pred = None

        if api_choice == 'MLflow':
            st.write(MLFLOW_URI)
            #st.write(data)
            pred = request_prediction(MLFLOW_URI, data)#[0] * 100000
            st.write(pred)
            st.write(pred["prediction"])
        elif api_choice == 'Cortex':
            pred = request_prediction(CORTEX_URI, data)[0] * 100000
        elif api_choice == 'Ray Serve':
            pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
        st.write(
            'Le prix médian d\'une habitation est de {:.2f}'.format(pred["prediction"]))
            #'Le prix médian d\'une habitation est de {:.2f}'.format(pred["prediction"][0]))
        
  


if __name__ == '__main__':
    main()
