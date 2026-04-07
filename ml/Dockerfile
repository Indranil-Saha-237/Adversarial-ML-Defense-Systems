# Use a lightweight base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app
# Install the required dependencies for ART and ML
 RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

#Install the required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML code and pre-trained model into the container
COPY app.py .

# Expose the port for the Flask API
EXPOSE 5001

# Start ML api
CMD ["python", "app.py"]