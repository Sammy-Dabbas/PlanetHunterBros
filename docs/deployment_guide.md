# Exoplanet Detection System - Google Cloud Deployment Guide

This guide will walk you through deploying your exoplanet detection system to Google Cloud Platform.

## Prerequisites

1. **Google Cloud Account**: Create an account at https://cloud.google.com/
2. **Google Cloud SDK**: Install from https://cloud.google.com/sdk/docs/install
3. **Project Setup**: Create a new Google Cloud project

## Deployment Options

We support two deployment methods:

### Option 1: Google Cloud Run (Recommended)
- Serverless, auto-scaling
- Pay only for what you use
- Easier setup
- Better for variable traffic

### Option 2: Google App Engine
- Managed platform
- More configuration options
- Better for consistent traffic

---

## Option 1: Deploy to Google Cloud Run

### Step 1: Install and Configure gcloud CLI

```bash
# Install Google Cloud SDK (if not already installed)
# Download from: https://cloud.google.com/sdk/docs/install

# Initialize gcloud
gcloud init

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Enable Required APIs

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build API
gcloud services enable cloudbuild.googleapis.com
```

### Step 3: Build and Deploy

```bash
# Navigate to your project directory
cd C:\Users\sammy\Desktop\NasaSpaceApps

# Build the container image using Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/exoplanet-detector

# Deploy to Cloud Run
gcloud run deploy exoplanet-detector \
  --image gcr.io/YOUR_PROJECT_ID/exoplanet-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

### Step 4: Access Your Application

After deployment, Cloud Run will provide a URL like:
```
https://exoplanet-detector-xxxxx-uc.a.run.app
```

Visit this URL to access your exoplanet detection system!

---

## Option 2: Deploy to Google App Engine

### Step 1: Install and Configure gcloud CLI

```bash
# Initialize gcloud (if not already done)
gcloud init

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Enable App Engine

```bash
# Create App Engine application (first time only)
gcloud app create --region=us-central
```

### Step 3: Deploy Application

```bash
# Navigate to your project directory
cd C:\Users\sammy\Desktop\NasaSpaceApps

# Deploy to App Engine
gcloud app deploy app.yaml

# When prompted, confirm deployment by typing 'Y'
```

### Step 4: Access Your Application

```bash
# Open your deployed app in browser
gcloud app browse
```

Your app will be available at:
```
https://YOUR_PROJECT_ID.uc.r.appspot.com
```

---

## Configuration Notes

### Memory and CPU Settings

For **Cloud Run**, the deployment uses:
- **Memory**: 2GB (handles model training and predictions)
- **CPU**: 2 vCPUs (for faster processing)
- **Timeout**: 300 seconds (5 minutes for long-running requests)

For **App Engine**, the `app.yaml` uses:
- **Instance class**: F2 (512MB memory, 1.2GHz CPU)
- **Auto-scaling**: 1-10 instances based on CPU utilization

### Adjusting Resources

If you need more resources, modify:

**Cloud Run:**
```bash
gcloud run deploy exoplanet-detector \
  --memory 4Gi \
  --cpu 4
```

**App Engine:** Edit `app.yaml`:
```yaml
instance_class: F4  # Upgrade to F4 (1GB) or F4_1G
```

---

## Cost Estimates

### Cloud Run (Recommended)
- **Free tier**: 2 million requests/month
- **Estimated cost**: $5-20/month for moderate usage
- **Best for**: Variable traffic, demos

### App Engine
- **Free tier**: 28 instance hours/day
- **Estimated cost**: $10-50/month with 1 always-on instance
- **Best for**: Consistent traffic

---

## Monitoring and Logs

### View Logs

**Cloud Run:**
```bash
gcloud run services logs read exoplanet-detector --limit 100
```

**App Engine:**
```bash
gcloud app logs tail -s default
```

### Web Console

Visit https://console.cloud.google.com/ to:
- Monitor traffic and performance
- View detailed logs
- Set up alerts
- Check resource usage

---

## Updating Your Application

### Cloud Run
```bash
# Rebuild and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/exoplanet-detector
gcloud run deploy exoplanet-detector \
  --image gcr.io/YOUR_PROJECT_ID/exoplanet-detector
```

### App Engine
```bash
# Simply redeploy
gcloud app deploy app.yaml
```

---

## Troubleshooting

### Issue: "Permission denied" errors
**Solution**: Ensure you have the necessary IAM roles:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/run.admin"
```

### Issue: Container build fails
**Solution**: Check Dockerfile syntax and ensure all files exist:
```bash
# Test Docker build locally first
docker build -t exoplanet-detector .
```

### Issue: Application crashes on startup
**Solution**: Check import paths are correct for the new `src/` structure in `app.py`

### Issue: Out of memory errors
**Solution**: Increase memory allocation or optimize model loading

---

## Security Recommendations

1. **Authentication**: Add authentication for production use
2. **API Keys**: Store NASA API keys in Secret Manager
3. **Rate Limiting**: Implement request throttling
4. **HTTPS**: Always enabled by default on Cloud Run/App Engine

---

## Next Steps

1. **Custom Domain**: Map your own domain name
2. **CI/CD**: Set up automatic deployments with Cloud Build triggers
3. **Monitoring**: Configure Cloud Monitoring alerts
4. **Scaling**: Adjust auto-scaling parameters based on usage

---

## Support and Resources

- **Google Cloud Documentation**: https://cloud.google.com/docs
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **App Engine Docs**: https://cloud.google.com/appengine/docs
- **Pricing Calculator**: https://cloud.google.com/products/calculator

---

## Quick Reference Commands

```bash
# Cloud Run - Deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/exoplanet-detector
gcloud run deploy exoplanet-detector --image gcr.io/YOUR_PROJECT_ID/exoplanet-detector --platform managed --region us-central1 --allow-unauthenticated

# App Engine - Deploy
gcloud app deploy app.yaml

# View logs
gcloud run services logs read exoplanet-detector
gcloud app logs tail

# Open in browser
gcloud run services describe exoplanet-detector --format="value(status.url)"
gcloud app browse
```

---

**Good luck with your deployment! Your exoplanet detection system will be accessible to judges worldwide.**
