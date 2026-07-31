# Minimal test runner image for SpidersSentiments
FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps (for building wheels and Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install only what's needed for tests (lightweight)
COPY requirements_core.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy source
COPY . .

# Default command runs the test suite
CMD ["pytest", "-q"]
