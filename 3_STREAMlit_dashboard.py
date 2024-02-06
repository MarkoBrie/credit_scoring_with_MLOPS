import pandas as pd
import streamlit as st
import requests



def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    st.write(data)
    #data_json = {'data': data}
    data_json = data
    
    #data_json = {'dataframe_split': data.to_dict(orient='records')}
    st.write(data_json)
    
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    MLFLOW_URI = 'http://127.0.0.1:8099/invocations'
    CORTEX_URI = 'http://0.0.0.0:8890/'
    RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['MLflow', 'Cortex', 'Ray Serve'])

    st.title('Median House Price Prediction')

    revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
                                 min_value=0., value=3.87, step=1.)

    age_med = st.number_input('Âge médian des maisons dans le secteur',
                              min_value=0., value=28., step=1.)

    nb_piece_med = st.number_input('Nombre moyen de pièces',
                                   min_value=0., value=5., step=1.)

    nb_chambre_moy = st.number_input('Nombre moyen de chambres',
                                     min_value=0., value=1., step=1.)

    taille_pop = st.number_input('Taille de la population dans le secteur',
                                 min_value=0, value=1425, step=100)

    occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
                                     min_value=0., value=3., step=1.)

    latitude = st.number_input('Latitude du secteur',
                               value=35., step=1.)

    longitude = st.number_input('Longitude du secteur',
                                value=-119., step=1.)

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
            st.write(data)
            pred = request_prediction(MLFLOW_URI, data)#[0] * 100000
        elif api_choice == 'Cortex':
            pred = request_prediction(CORTEX_URI, data)[0] * 100000
        elif api_choice == 'Ray Serve':
            pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
        st.write(
            'Le prix médian d\'une habitation est de {:.2f}'.format(pred["predictions"][0]))
  


if __name__ == '__main__':
    main()
