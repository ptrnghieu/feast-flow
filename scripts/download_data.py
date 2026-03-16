import opendatasets as od
import pandas as pd
import os
from pathlib import Path

def download_telco_data():
    """Download and extract the Telco Churn dataset"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from Kaggle
    dataset_url = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
    
    print("Downloading Telco Customer Churn dataset...")
    od.download(dataset_url, data_dir=str(data_dir))
    
    # Find the CSV file
    csv_path = data_dir / "telco-customer-churn" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if csv_path.exists():
        print(f"Dataset downloaded successfully to {csv_path}")
        return csv_path
    else:
        raise FileNotFoundError("Dataset file not found after download")

if __name__ == "__main__":
    download_telco_data()