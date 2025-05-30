FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY export_model.py .
COPY app.py .
COPY data_validation.py .

# Copy model files
# COPY best_resnext.pth .
# Download the model from Dropbox
RUN apt-get update && apt-get install -y wget && \
    wget -O best_resnext.pth "https://www.dropbox.com/scl/fi/eugcs5a9g017kr1tjqac1/best_resnext.pth?rlkey=6ma68mxoxwquv0sm4enad0ucf&st=2yrfhcft&dl=1" && \
    apt-get remove -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY exported_model/ ./exported_model/
RUN python export_model.py

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