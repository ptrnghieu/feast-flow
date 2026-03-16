import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataTransformer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load the raw dataset"""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self
        
    def handle_missing_values(self):
        """Handle missing values and data cleaning"""
        # Convert TotalCharges to numeric, coerce errors to NaN
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (for new customers)
        self.df['TotalCharges'].fillna(0, inplace=True)
        
        # Drop customerID missing values if any
        self.df.dropna(subset=['customerID'], inplace=True)
        
        return self
        
    def create_features(self):
        """Create engineered features"""
        # Create tenure ratio
        max_tenure = self.df['tenure'].max()
        self.df['customer_tenure_ratio'] = self.df['tenure'] / max_tenure
        
        # Create monthly charge average (handle division by zero)
        self.df['monthly_charge_avg'] = np.where(
            self.df['tenure'] > 0,
            self.df['TotalCharges'] / self.df['tenure'],
            self.df['MonthlyCharges']
        )
        
        # Create total services count
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        self.df['total_services'] = self.df[services].apply(
            lambda x: (x != 'No') & (x != 'No internet service')).sum(axis=1)
        
        return self
        
    def encode_categorical(self):
        """Encode categorical variables"""
        # Binary encoding
        binary_mapping = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
        binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                         'PaperlessBilling', 'Churn']
        
        for col in binary_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(binary_mapping).fillna(self.df[col])
        
        # One-hot encoding for multi-category features
        categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']
        self.df = pd.get_dummies(self.df, columns=categorical_columns, prefix=categorical_columns)
        
        return self
        
    def add_timestamps(self):
        """Add synthetic timestamps for Feast demonstration"""
        # Create synthetic event timestamps over the last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        # Generate random timestamps for each customer
        random_days = np.random.randint(0, 730, len(self.df))
        self.df['event_timestamp'] = [start_date + timedelta(days=int(x)) for x in random_days]
        
        # Create created timestamp (same as event timestamp for this demo)
        self.df['created_timestamp'] = self.df['event_timestamp']
        
        return self
        
    def select_final_features(self):
        """Select final features for Feast"""
        # Define feature columns we want to use
        feature_columns = [
            'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'customer_tenure_ratio', 'monthly_charge_avg', 'total_services',
            'gender', 'Partner', 'Dependents', 'SeniorCitizen',
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
            'event_timestamp', 'created_timestamp'
        ]
        
        # Target variable
        target_column = ['Churn']
        
        # Ensure all columns exist
        available_columns = [col for col in feature_columns + target_column if col in self.df.columns]
        self.df = self.df[available_columns]
        
        return self
        
    def save_processed_data(self):
        """Save the processed data"""
        output_path = "data/processed/telco_churn_processed.parquet"
        self.df.to_parquet(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return output_path

def main():
    """Main transformation function"""
    transformer = DataTransformer("data/raw/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    transformer.load_data()\
              .handle_missing_values()\
              .create_features()\
              .encode_categorical()\
              .add_timestamps()\
              .select_final_features()\
              .save_processed_data()
    
    print("Data transformation completed successfully!")

if __name__ == "__main__":
    main()