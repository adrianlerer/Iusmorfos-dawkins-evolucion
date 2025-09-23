# Dockerfile for Iusmorfos: Reproducible Legal Evolution Analysis
# Based on gold-standard reproducibility practices

FROM python:3.8-slim-bullseye

# Metadata following OCI standards
LABEL org.opencontainers.image.title="Iusmorfos: Dawkins Legal Evolution"
LABEL org.opencontainers.image.description="Reproducible environment for legal system evolution analysis"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Adrian Lerer <adrian@example.com>"
LABEL org.opencontainers.image.source="https://github.com/adrianlerer/Iusmorfos-dawkins-evolucion"
LABEL org.opencontainers.image.licenses="MIT"

# Set reproducibility environment variables
ENV PYTHONHASHSEED=42
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.lock requirements.txt ./

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir --upgrade pip==23.2.1 && \
    pip install --no-cache-dir -r requirements.lock

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p results outputs logs data/processed

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys, numpy, pandas, matplotlib; print('Environment OK')"

# Default command: run tests to validate environment
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# Alternative entrypoints for different use cases:
# Run main experiment: docker run iusmorfos python src/experimento_piloto_biomorfos.py
# Interactive shell: docker run -it iusmorfos /bin/bash
# Jupyter notebook: docker run -p 8888:8888 iusmorfos jupyter notebook --ip=0.0.0.0 --allow-root