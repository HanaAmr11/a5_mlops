FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

COPY . /app

RUN echo "Preparing container for model run: ${RUN_ID}"

CMD echo "Downloading model for Run ID: ${RUN_ID}" && echo "Model download simulated successfully"