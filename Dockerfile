FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY export_model.py .
COPY app.py .
COPY data_validation.py .

# Copy model files
COPY best_resnext.pth .
COPY exported_model/ ./exported_model/

# Create directories for MLflow and metrics
RUN mkdir -p mlruns
RUN mkdir -p metrics

# Expose ports for FastAPI (8080) and Prometheus metrics (8000)
EXPOSE 8080
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=exported_model/model.pt
ENV PYTHONUNBUFFERED=1

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]