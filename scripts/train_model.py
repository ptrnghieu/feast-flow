import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle
import json
from datetime import datetime
from pathlib import Path

def get_training_data():
    """Get training data using Feast offline store"""
    from feast import FeatureStore
    
    # Initialize feature store
    store = FeatureStore(repo_path="feature_repo/")
    
    # Get list of customer entities
    entity_df = pd.read_parquet("data/processed/telco_churn_processed.parquet")
    entity_df = entity_df[['customerID', 'event_timestamp']].rename(columns={'customerID': 'customerID'})
    
    # Get training data from Feast
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service("churn_service_v1")
    ).to_df()
    
    # Merge with target variable
    target_df = pd.read_parquet("data/processed/telco_churn_processed.parquet")
    target_df = target_df[['customerID', 'Churn']].rename(columns={'customerID': 'customerID'})
    
    training_df = training_df.merge(target_df, on='customerID', how='left')
    
    return training_df

def train_churn_model():
    """Train the churn prediction model"""
    print("Loading training data from Feast...")
    df = get_training_data()
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['customerID', 'event_timestamp', 'Churn']]
    X = df[feature_columns]
    y = df['Churn']
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"Feature count: {len(feature_columns)}")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "churn_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    feature_info = {
        'feature_names': feature_columns,
        'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
        'training_date': datetime.now().isoformat(),
        'accuracy': accuracy,
        'classification_report': report
    }
    
    with open(model_dir / "model_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"Model saved to {model_path}")
    return model, feature_columns, accuracy

if __name__ == "__main__":
    train_churn_model()