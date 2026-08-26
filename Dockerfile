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

# Copy application source code
COPY src/ ./src/

# Use ENTRYPOINT so users can pass CLI arguments directly to the container
ENTRYPOINT ["python", "src/main.py"]
