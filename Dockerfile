# Production Multi-Stage Dockerfile for Multi-Modal Vision Transformer

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy repository files
COPY . .

# Create output and checkpoint directories
RUN mkdir -p checkpoints outputs data

# Expose Gradio default port
EXPOSE 7860

# Default command launches Gradio application
CMD ["python", "app/gradio_app.py"]
