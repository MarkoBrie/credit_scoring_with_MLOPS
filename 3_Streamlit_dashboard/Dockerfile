# python base image in the container from Docker Hub
FROM python:3.8-bookworm

# copy files to the /app folder in the container
COPY ./3_Streamlit_dashboard/3_STREAMlit_dashboard.py /app/3_STREAMlit_dashboard.py
COPY ./3_Streamlit_dashboard/data /app/data


# set the working directory in the container to be /app
WORKDIR /app

# install the packages from the Pipfile in the container

RUN pip install pipenv
RUN pip install pandas
RUN pip install streamlit
RUN pip install streamlit_option_menu
RUN pip install requests
RUN pip install matplotlib
RUN pip install seaborn 
RUN pip install bokeh==2.4.3
RUN pip install plotly

# expose the port that uvicorn will run the app on
ENV PORT=8501
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "3_STREAMlit_dashboard.py"]