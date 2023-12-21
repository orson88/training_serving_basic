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

# Expose port 8080
EXPOSE 8080

# Command to start the FastAPI app using Uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]