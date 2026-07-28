# Use a lightweight, official Python runtime base image
FROM python:3.12-slim

# Set the working directory inside the container runtime
WORKDIR /app

# Copy only the requirements first to take advantage of Docker's layer caching
COPY requirements.txt .

# Install the dependencies without saving bloated cache files
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files into the container image
COPY . .

# Render primarily looks for port 10000 by default, though we bind dynamically
EXPOSE 10000

# Fire up Gunicorn and bind it to the environment's dynamic port variable
CMD gunicorn --bind 0.0.0.0:$PORT app:app