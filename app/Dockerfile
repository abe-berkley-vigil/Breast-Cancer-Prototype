# Use the official Python image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Upgrade pip
RUN pip install --upgrade pip

# Copy all required files to the docker image.
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_classification/ breast_cancer_classification/
COPY models/ models/
COPY data/ data/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]