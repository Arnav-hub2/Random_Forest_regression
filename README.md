# Loan Amount Prediction App

This is a machine learning application that predicts loan amounts using a Random Forest Regression model trained on loan approval data.

## Project Structure

```
├── rf_reg.ipynb                 # Jupyter notebook with model exploration
├── pipeline.py                  # Script to train and save the model
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── loan_approval_dataset.csv    # Dataset
├── models/                      # Directory containing saved model artifacts
│   ├── rf_model.pkl            # Trained Random Forest model
│   ├── label_encoders.pkl      # Categorical variable encoders
│   └── feature_names.pkl       # Feature column names
└── README.md                    # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the pipeline to train the model and save the artifacts:

```bash
python pipeline.py
```

This will:
- Load the loan approval dataset
- Preprocess and encode categorical variables
- Train a Random Forest Regression model
- Save the model, encoders, and feature names to the `models/` directory

### 3. Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Deploying to Streamlit Cloud

### Prerequisites
- GitHub account with the repository pushed
- Streamlit Cloud account (free at https://share.streamlit.io)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Add loan prediction app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with your GitHub account
   - Click "New app"
   - Select your GitHub repo, branch, and file path (app.py)
   - Click "Deploy"

### Important Notes for Streamlit Cloud

- Ensure the `loan_approval_dataset.csv` file is in your repository
- Make sure the `models/` directory with saved artifacts is in your repository
- If the model doesn't exist, Streamlit Cloud will fail. Ensure you run `pipeline.py` locally and commit the `models/` folder

## Features

### Prediction Tab
- Input borrower information (dependents, income, CIBIL score)
- Enter loan details (term, residential/commercial/luxury/bank assets)
- Specify education level, employment status, and loan status
- Get predicted loan amount with loan-to-income ratio

### Model Info Tab
- View model architecture and parameters
- See all features used in the model
- Check categorical variable encodings

## Model Performance

- **R² Score:** 0.8515 (explains 85.15% of variance)
- **Mean Absolute Error:** ₹2,591,401.64
- **Mean Squared Error:** 1.19e+13

## Input Features

1. `loan_id` - Loan identifier
2. ` no_of_dependents` - Number of dependents
3. ` education` - Education level (Graduate/Not Graduate)
4. ` self_employed` - Employment status (Yes/No)
5. ` income_annum` - Annual income
6. ` loan_term` - Loan duration in months
7. ` cibil_score` - Credit score
8. ` residential_assets_value` - Residential property value
9. ` commercial_assets_value` - Commercial property value
10. ` luxury_assets_value` - Luxury assets value
11. ` bank_asset_value` - Bank savings value
12. ` loan_status` - Loan approval status

## Troubleshooting

### Model files not found
- Run `python pipeline.py` to generate the model files
- Ensure the `models/` directory is created with all three pickle files

### Import errors
- Reinstall requirements: `pip install -r requirements.txt --upgrade`
- Check Python version (3.8+)

### Prediction errors
- Verify input data types match the expected format
- Check that categorical variables use the exact values shown in the dropdown

## Future Improvements

- Add more features for better predictions
- Implement model versioning
- Add prediction confidence intervals
- Create API endpoints
- Add user feedback mechanism

## License

This project is for educational purposes.
