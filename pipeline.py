import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def train_pipeline(data_path='loan_approval_dataset.csv'):
    """
    Train the Random Forest Regression model pipeline and save artifacts
    """
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    print("Data shape:", data.shape)
    print("Data columns:", data.columns.tolist())
    
    # Prepare features and target
    X = data.drop(' loan_amount', axis=1)
    y = data[' loan_amount']
    
    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    print(f"Categorical columns encoded: {categorical_cols}")
    
    # Train Random Forest model
    print("Training Random Forest Regression model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model and encoders
    print("\nSaving model and encoders...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    # Save label encoders
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model saved to models/rf_model.pkl")
    print("Label encoders saved to models/label_encoders.pkl")
    print("Feature names saved to models/feature_names.pkl")
    
    return rf, label_encoders, feature_names, {'mae': mae, 'mse': mse, 'r2': r2}

if __name__ == "__main__":
    train_pipeline()
