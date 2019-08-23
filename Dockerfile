#Use official Python runtime as a parent image
FROM python:3.7

WORKDIR /app

# Copy current directory to container at /app
COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World


