# Use an official Python runtime as a parent image
# 3.13 is new, sticking to 3.11/3.12 is safer for ML libraries usually, but let's match local if possible.
# Using slim to keep size down.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# No need for venv in container, system install is fine
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the content of the local src directory to the working directory
COPY . .

# Expose port 8080 (Standard for Cloud Run / App Engine)
EXPOSE 8080

# Run app.py using Gunicorn
# 1 worker usually fine for simple model, but can scale.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
