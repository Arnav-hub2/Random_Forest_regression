# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available)
- Git installed on your machine
- Repository pushed to GitHub

## Step-by-Step Deployment

### 1. Prepare Your Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Add loan prediction streamlit app"

# Push to GitHub (replace with your repo URL)
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Visit Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with your GitHub account
   
2. **Create New App:**
   - Click "New app" button
   - Select your GitHub repository
   - Choose the branch (usually "main")
   - Set main file path to: `app.py`
   - Click "Deploy"

3. **Wait for Deployment:**
   - Streamlit will automatically install dependencies from `requirements.txt`
   - The app will be deployed and live within 1-2 minutes

### 3. Share Your App

Once deployed, you'll get a shareable URL like:
```
https://your-username-loan-prediction-app.streamlit.app
```

## Critical Files for Deployment

Make sure these files are in your GitHub repository:

✅ `app.py` - Main Streamlit application
✅ `pipeline.py` - Model training script
✅ `requirements.txt` - Python dependencies
✅ `loan_approval_dataset.csv` - Training data
✅ `models/` folder with:
   - `rf_model.pkl`
   - `label_encoders.pkl`
   - `feature_names.pkl`
✅ `.streamlit/config.toml` - Streamlit configuration
✅ `README.md` - Documentation

## Important Notes

### Model Files Must Be Committed
The `models/` directory with all three `.pkl` files must be committed to GitHub:
```bash
git add models/
git commit -m "Add trained model files"
git push
```

### If Running Pipeline on Streamlit Cloud
Instead of using pickle files, you can embed the model training in the Streamlit app with caching:
```python
@st.cache_resource
def load_or_train_model():
    # Check if model exists locally
    if os.path.exists('models/rf_model.pkl'):
        # Load saved model
        ...
    else:
        # Train new model
        ...
```

## Troubleshooting Deployment

### "Model files not found" Error
**Solution:**
- Ensure `models/` folder is committed to GitHub
- Run `pipeline.py` locally first
- Commit all pickle files to GitHub

### App takes too long to load
**Solution:**
- Optimize model loading with `@st.cache_resource`
- Reduce dataset size if possible
- Check for heavy computations in the app

### "ModuleNotFoundError" for dependencies
**Solution:**
- Update `requirements.txt` with correct versions
- Push changes to GitHub
- Reboot the app from Streamlit Cloud dashboard

### App still loading but nothing appears
**Solution:**
- Check Streamlit Cloud logs
- Verify app.py has correct file paths
- Ensure all imports work (run locally first)

## Monitoring Your Deployment

1. **Check App Status:**
   - Visit your app URL
   - Look for any error messages in the browser console

2. **View Logs:**
   - Go to https://share.streamlit.io
   - Click on your app
   - View deployment logs

3. **Update Your App:**
   - Make changes locally
   - Push to GitHub
   - Streamlit will auto-redeploy

## Environment Variables (Optional)

If you need to use environment variables:

1. Go to your app settings on Streamlit Cloud
2. Add secrets in the "Secrets" section
3. Access in your app:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

## Cost Information

- **Free Tier:**
  - 1 app per month
  - Limited resources
  - Suitable for small projects

- **Pro Tier:**
  - More apps
  - Priority support
  - Enhanced resources

Visit https://streamlit.io/pricing for current pricing.

## Example Deployment Checklist

- [ ] All code tested locally
- [ ] `requirements.txt` updated with exact versions
- [ ] `models/` folder with `.pkl` files exists
- [ ] `loan_approval_dataset.csv` in repository
- [ ] `.gitignore` configured
- [ ] Code pushed to GitHub
- [ ] GitHub auth connected to Streamlit Cloud
- [ ] Correct repo/branch/file selected
- [ ] App deployed and accessible

## Getting Help

- **Streamlit Docs:** https://docs.streamlit.io
- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-cloud
- **Community Forum:** https://discuss.streamlit.io
