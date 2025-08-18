# Multi-stage build for optimized image size
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS base

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models logs GradioTEMP/chunks_analysis \
    && chmod -R 777 GradioTEMP logs

# Copy application code
COPY config ./config
COPY core ./core
COPY utils ./utils
COPY docs ./docs
COPY logos ./logos
COPY dashboard.py .
COPY env.example .

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SHARE=false
ENV SERVER_NAME=0.0.0.0
ENV SERVER_PORT=7877

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7877/api/health')" || exit 1

# Expose port
EXPOSE 7877

# Run the application
CMD ["python", "dashboard.py"] 