from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
import json
from pathlib import Path

# Feast imports
from feast import FeatureStore

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Real-time churn prediction using Feast feature store",
    version="1.0.0"
)

# Load model
model_path = Path("model/churn_model.pkl")
model_info_path = Path("model/model_info.json")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    feature_names = model_info['feature_names']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []

# Initialize Feast feature store
store = FeatureStore(repo_path="feature_repo")

class PredictionRequest(BaseModel):
    customerID: str

class PredictionResponse(BaseModel):
    customerID: str
    churn_prediction: bool
    churn_probability: float
    features_used: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feast_connected: bool

@app.get("/")
async def root():
    return {"message": "Telco Churn Prediction API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    feast_status = True
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        feast_connected=feast_status
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Make churn prediction for a customer"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get features from Feast online store
        feature_vector = store.get_online_features(
            features=store.get_feature_service("churn_service_v1"),
            entity_rows=[{"customerID": request.customerID}]
        ).to_dict()
        
        # Prepare features for model
        features = []
        for feature in feature_names:
            # Get the feature value from Feast response
            value = feature_vector[feature][0]
            if value is None:
                value = 0  # Default value for missing features
            features.append(value)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]
        
        return PredictionResponse(
            customerID=request.customerID,
            churn_prediction=bool(prediction),
            churn_probability=float(probability),
            features_used=feature_names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features/{customerID}")
async def get_customer_features(customerID: str):
    """Get feature values for a customer from Feast"""
    try:
        feature_vector = store.get_online_features(
            features=store.get_feature_service("churn_service_v1"),
            entity_rows=[{"customerID": customerID}]
        ).to_dict()
        
        # Clean up the response
        cleaned_features = {}
        for key, value in feature_vector.items():
            cleaned_features[key] = value[0] if value else None
        
        return cleaned_features
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching features: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)