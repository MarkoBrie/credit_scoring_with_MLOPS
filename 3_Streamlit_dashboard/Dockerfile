# python base image in the container from Docker Hub
FROM python:3.8-bookworm

# copy files to the /app folder in the container
COPY ./main.py /app/main.py
COPY ./Pipfile /app/Pipfile
COPY ./Pipfile.lock /app/Pipfile.lock

# set the working directory in the container to be /app
WORKDIR /app

# install the packages from the Pipfile in the container
RUN pip install pipenv
# Install uvicorn
RUN pip install pandas
RUN pip install streamlit
RUN pip install request

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000

# Run app.py when the container launches
CMD ["streamlit", "run", "3_STREAMLIT_dashboard.py"]