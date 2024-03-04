import pandas as pd
import streamlit as st
import requests

def request_prediction(model_uri: str, data: dict) -> dict:
    """
    Function to request a prediction from a deployed model.

    Args:
        model_uri (str): The URI of the deployed model.
        data (dict): The input data for which prediction is requested.

    Returns:
        dict: The prediction result in JSON format.
        
    Raises:
        Exception: If the request to the model fails.
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def hight_of_selected_point(hist, data, highlighted_index):
    bin_counts = [rect.get_height() for rect in hist.patches]
    print(len(bin_counts))
    print(len(bin_counts)/2)
    print(min(data['DAYS_BIRTH']), " ", max(data['DAYS_BIRTH']))
    print("selected point: ", data.loc[highlighted_index, 'DAYS_BIRTH'])
    scaled_point = int(round(data.loc[highlighted_index, 'DAYS_BIRTH']-min(data['DAYS_BIRTH'])))
    print("scaled point: ", scaled_point)

    steps = (max(data['DAYS_BIRTH'])-min(data['DAYS_BIRTH']))/(len(bin_counts)/2)
    print("steps :", steps )
    
    if data.loc[highlighted_index, 'TARGET'] == 0:
        bucket = int(round(scaled_point / steps,0))
 
    elif data.loc[highlighted_index, 'TARGET'] == 1:
        bucket = int(round(scaled_point / steps,0)+(len(bin_counts)/2))
        
    print("bucket :", bucket)
    hight = bin_counts[bucket]/2
    print("hight :", hight)
    
    return hight

def plot_histogram(data):

    # Highlighted data point
    highlighted_index = 0  # Index of the data point to highlight
    highlighted_value = data.loc[highlighted_index, 'DAYS_BIRTH']

    # Plotting
    fig, ax = plt.subplots()
    hist = sns.histplot(data=data, x='DAYS_BIRTH', hue='TARGET', kde=True,  multiple='stack', ax=ax) #stat='density',
    # Get the counts for each bin
    hight_P = hight_of_selected_point(hist, data, highlighted_index)

    # Highlight one specific data point
    if Y_train.loc[highlighted_index, 'TARGET'] == 1:
        ax.scatter(highlighted_value, hight_P, color='red', label='Highlighted Point', zorder=5)
    elif Y_train.loc[highlighted_index, 'TARGET'] == 0:
        ax.scatter(highlighted_value, hight_P, color='blue', label='Highlighted Point', zorder=5)

    # Customize plot
    ax.set_xlabel('Customer Age')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Stacked Distribution of Customer Age with Highlighted Point')
    legend = ax.get_legend()
    handles = legend.legend_handles
    legend.remove()
    ax.legend(handles, ['0 pays', '1 will have difficulty'], title='Client group')

    st.show(fig)
    


def main():
    MLFLOW_URI = 'https://fastapi-cd-webapp.azurewebsites.net/predict'
    #MLFLOW_URI = 'http://0.0.0.0:8000/predict'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['MLflow', 'Option 2', 'Option 3'])

    st.title('Prédiction du Credit Score avec ID')

    data_slice = pd.read_csv('data/X_train_slice.csv')

    plot_histogram(data_slice)

    ids_test = pd.read_csv('data/test_ids.csv')
    X_train = pd.read_csv('data/X_test.csv')
    feature_name = pd.read_csv('data/feature_names.csv')

    st.write('data size ', X_train.shape )
    # Set feature names as column names for X_train
    X_train.columns = feature_name['0'].tolist()
    selected_columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE', 'AMT_CREDIT']

    X_train["ID"] = ids_test
    X_train.set_index("ID", inplace=True)
    st.write(X_train.shape)

    # Select the desired columns from X_train
    selected_features = X_train[selected_columns]
    # Print the selected features
    st.write('selected features', selected_features)
    
    # Select columns with data type 'int64'
    int_columns = X_train.select_dtypes(include=['int64']).columns
    # Convert selected columns to int
    X_train[int_columns] = X_train[int_columns].astype('float')

    # Select columns with data type 'int64'
    int_columns = X_train.select_dtypes(include=['bool']).columns

    # Convert selected columns to int
    X_train[int_columns] = X_train[int_columns].astype('float')

    st.write(X_train.info())

    id_list = ids_test.iloc[:,0].values.tolist()

    selected_id = st.selectbox('Search and select an ID', options=id_list, index=0, format_func=lambda x: x if x else 'Search...')

    #get selected_id index in ids_test and use the index to get the data
    st.write(selected_id)
    st.write(X_train.loc[selected_id].shape)
    st.write(X_train.loc[selected_id].values.tolist())
    st.write(X_train.info())
    data =  { "data_point":X_train.loc[selected_id].values.tolist()}
    #st.write(data)
  
    selected_data = X_train.loc[selected_id, selected_columns]
    st.write('for client', selected_id)
    st.write(selected_data)

    predict_btn = st.button('Prédire')
    if predict_btn:
        """
        function that sends data to a model API and depicts the result
        """
        #st.write(data)
        st.write("after")
        lst = [0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    
        #data_list = [float(i) for i in lst]
        #data = { "data_point":[[0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]}
        #data = { "data_point": data_list}
                
        pred = None

        st.write(MLFLOW_URI)
        #st.write(data)
        pred = request_prediction(MLFLOW_URI, data)#[0] * 100000
        score = 0 if pred["prediction"] < 0.2 else 1
        #st.write(pred)
        st.write(pred["prediction"], " -> score ", score)

        st.write(
            'Le score crédit est de {:.2f}'.format(score))
            

if __name__ == '__main__':
    main()
