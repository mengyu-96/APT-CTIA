# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend requirements file into the container
COPY backend/requirements.txt /app/backend/requirements.txt

# Install any needed packages specified in requirements.txt
# Note: Installing torch and torch-geometric can be slow and large
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the backend code
COPY backend /app/backend
COPY utils /app/utils

# Create necessary directories for persistence
RUN mkdir -p /app/uploads \
    /app/results_archive \
    /app/dataset_TXT \
    /app/reports

# Set environment variables
ENV PYTHONPATH=/app:/app/backend
ENV FLASK_APP=backend/api.py

# Expose port 5001
EXPOSE 5001

# Run the API with a production WSGI server.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "300", "backend.wsgi:app"]
