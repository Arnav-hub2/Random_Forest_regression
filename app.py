import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Loan Amount Prediction",
    page_icon="💰",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'loan_approval_dataset.csv'
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'rf_model.pkl'
ENCODER_PATH = MODEL_DIR / 'label_encoders.pkl'
FEATURES_PATH = MODEL_DIR / 'feature_names.pkl'

@st.cache_resource
def load_or_train_model():
    """Load saved artifacts, or train them from the bundled dataset if needed."""
    if MODEL_PATH.exists() and ENCODER_PATH.exists() and FEATURES_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        return model, label_encoders, feature_names

    if not DATA_PATH.exists():
        st.error(
            'Model files are missing and the training dataset is not available. '
            'Make sure loan_approval_dataset.csv is committed to the repository.'
        )
        st.stop()

    data = pd.read_csv(DATA_PATH)
    X = data.drop(' loan_amount', axis=1)
    y = data[' loan_amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])
        label_encoders[col] = encoder

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    st.info(
        f'Model trained automatically because saved artifacts were missing. '
        f'R² score: {r2:.4f}, MAE: {mae:,.2f}'
    )
    return model, label_encoders, X.columns.tolist()


def load_model():
    """Compatibility wrapper for the rest of the app."""
    try:
        return load_or_train_model()
    except Exception as exc:
        st.error(f'Unable to load or train the model: {exc}')
        st.stop()

# Main app
def main():
    st.title("💰 Loan Amount Prediction App")
    st.markdown("---")
    
    # Load model
    model, label_encoders, feature_names = load_model()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "Model Info"])
    
    with tab1:
        st.header("Make a Prediction")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Borrower Information")
            loan_id = st.number_input("Loan ID", value=1, min_value=1)
            no_of_dependents = st.number_input("Number of Dependents", value=2, min_value=0, max_value=10)
            income_annum = st.number_input("Annual Income (₹)", value=500000, min_value=0, step=10000)
            cibil_score = st.number_input("CIBIL Score", value=750, min_value=300, max_value=900)
        
        with col2:
            st.subheader("Loan Details & Assets")
            loan_term = st.number_input("Loan Term (months)", value=60, min_value=12, max_value=480, step=12)
            residential_assets = st.number_input("Residential Assets Value (₹)", value=1000000, min_value=0, step=100000)
            commercial_assets = st.number_input("Commercial Assets Value (₹)", value=500000, min_value=0, step=100000)
            luxury_assets = st.number_input("Luxury Assets Value (₹)", value=200000, min_value=0, step=100000)
            bank_asset = st.number_input("Bank Asset Value (₹)", value=300000, min_value=0, step=100000)
        
        # Categorical inputs
        st.subheader("Education & Employment Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            education = st.selectbox("Education Level", [' Graduate', ' Not Graduate'])
        
        with col2:
            self_employed = st.selectbox("Self Employed", [' No', ' Yes'])
        
        with col3:
            loan_status = st.selectbox("Loan Status", [' Approved', ' Rejected'])
        
        st.markdown("---")
        
        # Make prediction
        if st.button("Predict Loan Amount", type="primary", use_container_width=True):
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'loan_id': [loan_id],
                    ' no_of_dependents': [no_of_dependents],
                    ' education': [education],
                    ' self_employed': [self_employed],
                    ' income_annum': [income_annum],
                    ' loan_term': [loan_term],
                    ' cibil_score': [cibil_score],
                    ' residential_assets_value': [residential_assets],
                    ' commercial_assets_value': [commercial_assets],
                    ' luxury_assets_value': [luxury_assets],
                    ' bank_asset_value': [bank_asset],
                    ' loan_status': [loan_status]
                })
                
                # Encode categorical variables
                for col in label_encoders:
                    if col in input_data.columns:
                        input_data[col] = label_encoders[col].transform(input_data[col])
                
                # Ensure column order matches training data
                input_data = input_data[feature_names]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.success("✅ Prediction Complete!")
                st.markdown("---")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric(
                        label="Predicted Loan Amount",
                        value=f"₹{prediction:,.2f}",
                        delta=None
                    )
                
                with col2:
                    st.info(f"**Annual Income:** ₹{income_annum:,}\n\n**Loan-to-Income Ratio:** {(prediction/income_annum)*100:.2f}%")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model Type:** Random Forest Regressor")
            st.write(f"**Number of Trees:** 100")
            st.write(f"**Random State:** 42")
        
        with col2:
            st.subheader("Features Used")
            st.write(f"**Total Features:** {len(feature_names)}")
            st.write("**Feature Names:**")
            for i, feat in enumerate(feature_names, 1):
                st.write(f"{i}. {feat}")
        
        st.markdown("---")
        st.subheader("Categorical Encodings")
        
        for col, encoder in label_encoders.items():
            st.write(f"**{col}:**")
            classes_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            for cls, code in classes_dict.items():
                st.write(f"  • {cls} → {code}")

if __name__ == "__main__":
    main()
