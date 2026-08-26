FROM python:3.10-slim

# Install system dependencies required by ffmpeg and opencv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and frontend
COPY src/ ./src/
COPY frontend/ ./frontend/

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose web server port
EXPOSE 8000

# Use ENTRYPOINT to dynamically handle CLI or Web UI requests
ENTRYPOINT ["./entrypoint.sh"]
