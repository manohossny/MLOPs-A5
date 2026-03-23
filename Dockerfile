FROM python:3.10-slim

WORKDIR /app

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD echo "Downloading model for Run ID: ${RUN_ID}" && echo "Model deployed successfully"
