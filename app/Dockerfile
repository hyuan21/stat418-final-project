FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

COPY app/ /app/

# Cloud Run sends traffic to whatever port the container listens on,
# passed in via the PORT environment variable (default 8080).
# Locally with docker-compose we set PORT=8501 via docker-compose.yml.
ENV PORT=8080
EXPOSE 8080

# Use shell form so $PORT is expanded at runtime
CMD streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
