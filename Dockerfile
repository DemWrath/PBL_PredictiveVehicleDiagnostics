# ============================================================================
# VEHICLE FAULT DETECTION API - DOCKER IMAGE
# ============================================================================
# Multi-stage build for smaller image size
# Production-ready with TensorFlow GPU support

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# DEPENDENCIES STAGE
# ============================================================================
FROM base as dependencies

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# PRODUCTION STAGE
# ============================================================================
FROM base as production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application files
COPY deploy_cloud_api.py .
COPY diagnostic_engine.py .
COPY lstm_classifier.py .
COPY config.py .
COPY classifier.h5 .

# Create directory for logs
RUN mkdir -p /app/logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# Run with Gunicorn (production server)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 deploy_cloud_api:app
