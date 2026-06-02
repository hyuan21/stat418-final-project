# Deployment Guide

This project has two deployable services:

1. **Flask API** → Google Cloud Run
2. **Streamlit App** → Streamlit Community Cloud

Both are free-tier friendly for course-project usage.

---

## 1. Deploy the Flask API to Google Cloud Run

### Prerequisites
- A Google Cloud project with billing enabled (the free tier covers this project's usage)
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Docker installed locally

### One-time setup
```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com \
                       artifactregistry.googleapis.com \
                       cloudbuild.googleapis.com
```

### Build, push, and deploy
From the project root:
```bash
# Build image from the API Dockerfile (note: build context is .)
gcloud builds submit \
    --tag gcr.io/YOUR_PROJECT_ID/nba-playoff-api \
    --config=- <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-f', 'api/Dockerfile', '-t', 'gcr.io/\$PROJECT_ID/nba-playoff-api', '.']
images: ['gcr.io/\$PROJECT_ID/nba-playoff-api']
EOF

# Or, simpler: build locally and push
docker build -f api/Dockerfile -t gcr.io/YOUR_PROJECT_ID/nba-playoff-api .
docker push gcr.io/YOUR_PROJECT_ID/nba-playoff-api

# Deploy
gcloud run deploy nba-playoff-api \
    --image gcr.io/YOUR_PROJECT_ID/nba-playoff-api \
    --platform managed \
    --region us-central1 \
    --memory 1Gi \
    --cpu 1 \
    --timeout 60 \
    --concurrency 80 \
    --allow-unauthenticated
```

Cloud Run prints the public URL when deployment completes. Test it (the current live URL for this project is shown below; substitute your own URL if you have redeployed):
```bash
curl https://nba-playoff-api-803317660037.us-central1.run.app/v1/health
```

### Automated CI/CD
The repo includes `.github/workflows/deploy.yml`. To enable it:
1. Create a service account in GCP with the **Cloud Run Admin** and **Storage Admin** roles
2. Download its JSON key
3. Add two GitHub Secrets in your repo:
   - `GCP_PROJECT_ID` — your project ID
   - `GCP_SA_KEY` — the entire JSON key file contents
4. Push to `main` — the workflow builds the Docker image and deploys it automatically

---

## 2. Deploy the Streamlit App to Streamlit Community Cloud

### One-time setup
1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account
2. Click **New app** → pick this repo, branch `main`, and the file `app/streamlit_app.py`

### Configure the API URL
On Streamlit Cloud, go to your app's **Settings → Secrets** and add:
```toml
API_URL = "https://nba-playoff-api-803317660037.us-central1.run.app"
```

Streamlit will redeploy automatically. The app reads `API_URL` from
`os.environ` (see `app/config.py`).

### Updating
Any push to `main` triggers an automatic redeploy on Streamlit Cloud — no
manual step needed.

---

## 3. Local end-to-end test (docker-compose)

```bash
docker-compose up --build
# API → http://localhost:8080
# App → http://localhost:8501
```

Use this to verify the full system works before deploying to the cloud.

---

## 4. Tearing things down (after course)

After the evaluation window (June 9, 2026):
```bash
# Delete Cloud Run service (stops all charges)
gcloud run services delete nba-playoff-api --region us-central1

# Delete container images
gcloud container images delete gcr.io/YOUR_PROJECT_ID/nba-playoff-api --force-delete-tags

# Streamlit Cloud: click "Delete" in the app dashboard
```
