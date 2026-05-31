FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY api/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Copy the project (both src/ and api/) into the container
COPY src/ /app/src/
COPY api/ /app/api/

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

WORKDIR /app/api
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 60 app:app
