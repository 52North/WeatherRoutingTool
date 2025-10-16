# Weather Routing Tool - Docker Image
FROM python:3.11-slim

# Install system dependencies required by Cartopy, GEOS, and PROJ
RUN apt-get update && apt-get install -y --no-install-recommends \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt requirements-without-deps.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -r requirements-without-deps.txt

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/logs

# Set environment variables
ENV WRT_FIGURE_PATH=/app/output
ENV PYTHONUNBUFFERED=1

# Run CLI as entry point
ENTRYPOINT ["wrt"]
CMD ["--help"]
