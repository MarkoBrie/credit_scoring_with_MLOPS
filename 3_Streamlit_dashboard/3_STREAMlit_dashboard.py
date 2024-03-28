import pandas as pd
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu

import requests

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider
from bokeh.transform import factor_cmap
from bokeh.palettes import Set3
import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.layouts import column

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def bokeh_scatter(data_slice, data_category, data_category_y):
    # Create a ColumnDataSource
    # Color for no risk
    data_slice['color'] = '#8dd3c7'

    # condition for RISK
    condition = data_slice['TARGET'] == 1

    # Color for RISK
    data_slice.loc[condition, 'color'] = '#fb8072'

    df = pd.DataFrame(
        {
            "gender": data_slice['CODE_GENDER'],
            "risk":  data_slice['TARGET'],
            "kpi1": data_slice[data_category],
            "kpi2": data_slice[data_category_y],
            "color": data_slice['color']
        }
    )

    source = bpl.ColumnDataSource.from_df(df)
    hover = bmo.HoverTool(
        tooltips=[
            ('gender', '@gender'),
            ("RISK", '@risk')
        ]
    )
    p = bpl.figure(tools=[hover, "pan", "wheel_zoom"],title='Scatter plot',
        x_axis_label=data_category,
        y_axis_label=data_category_y )

    p.scatter(
        'kpi1', 
        'kpi2', source=source, color='color')
    
    st.bokeh_chart(p, use_container_width=True)

#st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Risk Prediction', 'Dashboard','Scatter Plot', 'Customer View'], 
        icons=['house', "shield-check",'bar-chart', 'binoculars', 'gear'], menu_icon="cast", default_index=0)
    selected

api_choice = st.sidebar.selectbox(
        'Choose a model',
        ['LightGBM', 'XGB in progress', 'XGboost in progress'])

api_location = st.sidebar.selectbox(
        'Choose a model',
        ['AZURE', 'local'])
    
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
    print(data)
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

def plot_histogram(data, id, age, category):

    # Highlighted data point
    #highlighted_index = id  # Index of the data point to highlight
    #highlighted_value = data.loc[highlighted_index, category]

    # Plotting
    fig, ax = plt.subplots()
    hist = sns.histplot(data=data, x=category, hue='TARGET', kde=True,  multiple='stack', ax=ax) #stat='density',
    # Get the counts for each bin
    #hight_P = hight_of_selected_point(hist, data, highlighted_index)

    # Highlight one specific data point
    #if data.loc[highlighted_index, 'TARGET'] == 1:
    #    ax.scatter(highlighted_value, hight_P, color='red', label='Highlighted Point', zorder=5)
    #elif data.loc[highlighted_index, 'TARGET'] == 0:
    #    ax.scatter(highlighted_value, hight_P, color='blue', label='Highlighted Point', zorder=5)
    ax.scatter(age, 200, color='red', label='Selected Customer', zorder=5)

    # Customize plot
    ax.set_xlabel('Customer Age')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Stacked Distribution of Customer Age with Highlighted Point')
    legend = ax.get_legend()
    handles = legend.legend_handles
    legend.remove()
    ax.legend(handles, ['0 pays', '1 will have difficulty'], title='Client group')

    st.pyplot(fig)
    
def filter_dataframe(df: pd.DataFrame):
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
    return df

def main():
    MLFLOW_URI = 'https://fastapi-cd-webapp.azurewebsites.net/predict'
    MLFLOW_URI_local = 'http://0.0.0.0:8000/predict'
    
    if api_location == 'local' :
        MLFLOW_URI = MLFLOW_URI_local
    else:
        MLFLOW_URI = MLFLOW_URI


    ids_test = pd.read_csv('data/test_ids.csv')
    id_list = ids_test.iloc[:,0].values.tolist()

    X_train = pd.read_csv('data/X_test.csv')
    feature_name = pd.read_csv('data/feature_names.csv')
 
    # Set feature names as column names for X_train
    X_train.columns = feature_name['0'].tolist()
    
    selected_columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE', 'AMT_CREDIT']
    X_train['DAYS_BIRTH'] = abs(X_train['DAYS_BIRTH'])/365

    X_train["ID"] = ids_test
    X_train.set_index("ID", inplace=True)

    # Select columns with data type 'int64'
    int_columns = X_train.select_dtypes(include=['int64']).columns
    # Convert selected columns to int
    X_train[int_columns] = X_train[int_columns].astype('float')

    # Select columns with data type 'int64'
    int_columns = X_train.select_dtypes(include=['bool']).columns

    # Convert selected columns to int
    X_train[int_columns] = X_train[int_columns].astype('float')


    
    data_category = st.sidebar.selectbox(
        'Quelle donnée souhaitez vous étudier',
        ['age', 'Work_in_years', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE', 'AMT_CREDIT'])

    data_category_y = st.sidebar.selectbox(
        'Add an Y-axis', 
        ['age', 'Work_in_years', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE', 'AMT_CREDIT'],
        index=2
        )

    selected_id = st.sidebar.selectbox('Search and select an ID', options=id_list, index=0, format_func=lambda x: x if x else 'Search...')

    data =  { "data_point":X_train.loc[selected_id].values.tolist()}
    st.title('Prédiction du Credit Score')

    data_slice = pd.read_csv('data/X_train_slice.csv')
    #st.write(data_slice.shape)
    #st.write(data_slice)

    # 1 customer age
    age = pd.read_csv('data/1_age.csv')

    #selected_id = st.selectbox('Search and select an ID', options=id_list, index=0, format_func=lambda x: x if x else 'Search...')

    data =  { "data_point":X_train.loc[selected_id].values.tolist()}
    #st.write(data)
  
    selected_data = X_train.loc[selected_id, selected_columns]

    selected_client_DP = X_train.loc[selected_id, 'DAYS_BIRTH']

    columns = ['TARGET','age','Work_in_years','AMT_INCOME_TOTAL','AMT_CREDIT','CODE_GENDER']

    if selected == "Customer View":
        st.header('Customer View')
        # Create a row layout
        c1, c2= st.columns(2)

        with st.container():
            c1.write("c1")
            c2.write("c2")

        with c1:
            chart_data = age
            st.bar_chart(data=chart_data, x='Age',y=('TARGET0', 'TARGET1'))

        with c2:
            chart_data = age
            st.bar_chart(data=chart_data, x='Age',y=('AGE1ratio'))

        plot_histogram(data_slice, selected_id, selected_client_DP, data_category)


    elif selected == "Dashboard":
        c1, c2= st.columns(2)
        with st.container():
            c1.write("")
            c2.write("")
        with c1:
            selected_x = st.selectbox('Select X', options=columns, index=1, format_func=lambda x: x if x else 'Search...')
        with c2:
            selected_y = st.selectbox('Select Y', options=columns, index=3, format_func=lambda x: x if x else 'Search...')
        #st.write(data_slice)
        
        data_filter = filter_dataframe(data_slice)
        st.write(data_slice[['age','Work_in_years','AMT_INCOME_TOTAL','AMT_CREDIT','CODE_GENDER']].describe())
        st.write(data_filter[['age','Work_in_years','AMT_INCOME_TOTAL','AMT_CREDIT','CODE_GENDER']].describe())

        c1, c2, c3 = st.columns(3)
        with st.container():
            c1.write("")
            c2.write("")
            c3.write("")
        with c1:
            st.write(data_filter[['TARGET']].value_counts())
        with c2:
            st.write(data_filter[['CODE_GENDER']].value_counts())
        with c3:
            st.write(data_filter[['CODE_GENDER','TARGET']].value_counts())

        chart_data = pd.DataFrame(data_filter, columns=[selected_y, selected_x])
        st.scatter_chart(chart_data, x=selected_x, y= selected_y,)

    elif selected == "Scatter Plot":
        st.write("bokeh scatter here")
        c1, c2= st.columns(2)
        with st.container():
            c1.write("")
            c2.write("")
        with c1:
            selected_x = st.selectbox('Select X', options=columns, index=1, format_func=lambda x: x if x else 'Search...')
            st.write("x ", selected_x)
        with c2:
            selected_y = st.selectbox('Select Y', options=columns, index=3, format_func=lambda x: x if x else 'Search...')
            st.write("y ", selected_y)
        
        #chart_data = pd.DataFrame(data_slice, columns=[selected_y, selected_x, 'TARGET'])
        #st.scatter_chart(chart_data, x=selected_x, y= selected_y, color='TARGET')
        bokeh_scatter(data_slice, selected_x, selected_y)

    elif selected == "Risk Prediction":

        c1, c2= st.columns(2)
        with st.container():
            c1.write("")
            c2.write("")

        with c1:
            #c1.write('for client' + selected_id)
            c1.write(selected_data)
        with c2:
            predict_btn = st.button('Prédire')
            st.write("API endpoint at: ", MLFLOW_URI)
        
        
        if predict_btn:
            """
            Function that sends selected client data to a model API and depicts the result
            """

            # hard coded example data point
            #lst = [0, 0, 1, 1, 63000.0, 310500.0, 15232.5, 310500.0, 0.026392, 16263, -214.0, -8930.0, -573, 0.0, 1, 1, 0, 1, 1, 0, 2.0, 2, 2, 11, 0, 0, 0, 0, 1, 1, 0.0, 0.0765011930557638, 0.0005272652387098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
        
            #data_list = [float(i) for i in lst]
            #data = { "data_point":[[0.0, 0.0, 1.0, 0.0, 135000.0, 568800.0, 20560.5, 450000.0, 0.01885, 52.71506849315068, -2329.0, -5170.0, -812.0, 9.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7526144906031748, 0.7896543511176771, 0.1595195404777181, 0.066, 0.059, 0.9732, 0.7552, 0.0211, 0.0, 0.1379, 0.125, 0.2083, 0.0481, 0.0756, 0.0505, 0.0, 0.0036, 0.0672, 0.0612, 0.9732, 0.7648, 0.019, 0.0, 0.1379, 0.125, 0.2083, 0.0458, 0.0771, 0.0526, 0.0, 0.0011, 0.0666, 0.059, 0.9732, 0.7585, 0.0208, 0.0, 0.1379, 0.125, 0.2083, 0.0487, 0.0761, 0.0514, 0.0, 0.0031, 0.0392, 0.0, 0.0, 0.0, 0.0, -1740.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]}
            #data = { "data_point": data_list}
                    
            pred = None
            threshold = 0.2

            pred = request_prediction(MLFLOW_URI, data)
            score = 0 if pred["prediction"] < 0.2 else 1

            #st.write(pred["prediction"], " -> score ", score)
            #st.write('Le score crédit est de {:.2f}'.format(score))
            st.write("feature importance")
            st.write(pred['importance'])

            feature_importance_df = pd.DataFrame({'Feature': feature_name['0'].tolist(), 'Importance': pred['importance']})
            FI_sorted = feature_importance_df.sort_values(by=['Importance'], ascending=False)
            FI_sorted = FI_sorted[:20]
            st.write(FI_sorted)


            fig = go.Figure(go.Bar(
                x=FI_sorted['Importance'],
                y=FI_sorted['Feature'],
                orientation='h'))
            st.plotly_chart(fig)
            


            col1, col2, col3 = st.columns(3)
            col1.metric(label= "Score", value= score, delta=(str((score-threshold)/threshold)+" %"), delta_color="normal", help=None, label_visibility="visible")
            col2.metric("Probability", value=pred["prediction"])
            col3.metric("Threshold", "0.2")

            #st.header('Gauge chart')
            # Create a row layout
            #c1, c2= st.columns(2)

            #with st.container():
            #    c1.write("")
            #    c2.write("")

            #with c1:
            

            #with c2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = pred["prediction"],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "RISK PREDICTION", 'font': {'size': 24}},
                delta = {'reference': 0.2, 'increasing': {'color': "darkred"}},
                gauge = {
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': "gray"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.2], 'color': 'yellowgreen'},
                        {'range': [0.2, 1], 'color': 'mistyrose'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.2}}))

            fig.update_layout(paper_bgcolor = "white", font = {'color': "dimgray", 'family': "Arial"})
            st.plotly_chart(fig)
            
            st.write("In green is the area that depicts a low risk, whereas the red area shows predictions with risk.") 
            st.write("The red line is the threshold that is used to classify wether a client shows risk or not.")
                
            

if __name__ == '__main__':
    main()
