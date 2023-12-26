# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the poetry.lock and pyproject.toml to the working directory
COPY poetry.lock pyproject.toml /app/

# Install Poetry
RUN pip install poetry

# Install the project dependencies
RUN poetry install

# Copy the rest of the files to the working directory
COPY . /app


ENV MLFLOW_TRACKING_URI postgresql+psycopg2://postgres:postgres@mlflow-postgres/mlflow_db
ENV MLFLOW_S3_ENDPOINT_URL http://minio:9110
ENV AWS_ACCESS_KEY_ID abobusamogus
ENV AWS_SECRET_ACCESS_KEY darkmagapatriot
ENV INITIAL_DATA_PATH=/app/initial-data/
ENV PREPARED_DATA_PATH=/app/processed-data/

# Expose port 8080
EXPOSE 8080

# Command to start the FastAPI app using Uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]