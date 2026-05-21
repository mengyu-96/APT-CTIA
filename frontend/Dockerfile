# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the frontend requirements file
COPY frontend/requirements.txt /app/frontend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Copy the frontend code
COPY frontend /app/frontend

# Expose port 8501
EXPOSE 8501

# Set working directory to frontend for Streamlit
WORKDIR /app/frontend

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
