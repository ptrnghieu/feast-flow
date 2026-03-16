import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="FeastFlow - Feature Store",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class FeastFlowDashboard:
    def __init__(self, api_url="http://api:8000"):
        self.api_url = api_url
        self.customer_data = None
        
    def load_customer_data(self):
        """Load sample customer data for the demo"""
        try:
            self.customer_data = pd.read_parquet("data/processed/telco_churn_processed.parquet")
            return True
        except Exception as e:
            st.error(f"Error loading customer data: {e}")
            return False
    
    def check_api_health(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except:
            return False, None
    
    def get_prediction(self, customerID):
        """Get prediction from API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={"customerID": customerID},
                timeout=10
            )
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_features(self, customerID):
        """Get feature values from API"""
        try:
            response = requests.get(f"{self.api_url}/features/{customerID}", timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, response.json()
        except Exception as e:
            return False, {"error": str(e)}

def main():
    dashboard = FeastFlowDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 FeastFlow Demo</h1>', unsafe_allow_html=True)
    st.markdown("### End-to-End ML Pipeline with Feast Feature Store")
    
    # Check API health
    api_healthy, health_info = dashboard.check_api_health()
    
    if not api_healthy:
        st.error("🚨 API Service is not running. Please start the API first:")
        st.code("uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload")
        st.info("Once the API is running, refresh this page.")
        return
    
    # Load customer data
    if not dashboard.load_customer_data():
        return
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    st.sidebar.success(f"✅ API Status: {health_info['status']}")
    st.sidebar.info(f"📊 Model Loaded: {health_info['model_loaded']}")
    st.sidebar.info(f"🏪 Feast Connected: {health_info['feast_connected']}")
    
    # Customer selection
    st.sidebar.subheader("Customer Selection")
    customerIDs = dashboard.customer_data['customerID'].tolist()
    selected_customer = st.sidebar.selectbox(
        "Choose a customer:",
        customerIDs,
        index=0
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Customer Details")
        
        # Display customer information
        customer_info = dashboard.customer_data[
            dashboard.customer_data['customerID'] == selected_customer
        ].iloc[0]
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Tenure (months)", int(customer_info['tenure']))
            st.metric("Monthly Charges", f"${customer_info['MonthlyCharges']:.2f}")
            
        with info_col2:
            st.metric("Total Charges", f"${customer_info['TotalCharges']:.2f}")
            st.metric("Total Services", int(customer_info.get('total_services', 0)))
            
        with info_col3:
            actual_churn = "Yes" if customer_info.get('Churn', 0) == 1 else "No"
            st.metric("Actual Churn", actual_churn)
    
    # Prediction section
    with col2:
        st.subheader("🤖 Prediction")
        
        if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
            with st.spinner("Getting features from Feast and making prediction..."):
                # Simulate processing time for demo effect
                time.sleep(1)
                
                # Get prediction
                success, prediction_data = dashboard.get_prediction(selected_customer)
                
                if success:
                    # Display prediction results
                    churn_prob = prediction_data['churn_probability']
                    will_churn = prediction_data['churn_prediction']
                    
                    # Create gauge chart for churn probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = churn_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Probability %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction result
                    if will_churn:
                        st.error(f"🚨 High Risk of Churn ({churn_prob:.1%})")
                        st.write("This customer is likely to leave. Consider retention offers!")
                    else:
                        st.success(f"✅ Low Risk of Churn ({churn_prob:.1%})")
                        st.write("This customer is likely to stay.")
                        
                else:
                    st.error(f"Prediction failed: {prediction_data.get('detail', 'Unknown error')}")
    
    # Feature Store Section
    st.markdown("---")
    st.subheader("🏪 Feature Store - Live Features from Feast")
    
    if st.button("📥 Fetch Features from Feast Online Store", key="fetch_features"):
        with st.spinner("Retrieving real-time features from Feast..."):
            time.sleep(0.5)
            success, features_data = dashboard.get_features(selected_customer)
            
            if success:
                features_df = pd.DataFrame.from_dict(features_data, orient='index', columns=['Value'])
                features_df["Value"] = features_df["Value"].apply(lambda x: str(x) if x is not None else "")
                features_df = features_df[~features_df.index.str.contains('event_timestamp')]
                features_df = features_df[~features_df.index.str.contains('created_timestamp')]
                
                # st.write(features_df)

                # Define your feature groups
                demographics_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'total_services']
                financial_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'customer_tenure_ratio', 'monthly_charge_avg']
                contract_cols = [
                    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year'
                ]

                # Extract DataFrames for each group
                demographics_features = features_df.loc[features_df.index.isin(demographics_cols)]
                financial_features = features_df.loc[features_df.index.isin(financial_cols)]
                contract_features = features_df.loc[features_df.index.isin(contract_cols)]

                # Create 3 Streamlit columns
                col1, col2, col3 = st.columns(3)

                # 👥 Demographics
                with col1:
                    st.markdown("#### 👥 Demographics")
                    for feature, row in demographics_features.iterrows():
                        value = row["Value"]
                        if value is not None:
                            st.markdown(f"""
                            <div class="feature-card">
                                <strong>{feature}</strong><br>
                                {value}
                            </div>
                            """, unsafe_allow_html=True)

                # 💰 Financials
                with col2:
                    st.markdown("#### 💰 Financials")
                    for feature, row in financial_features.iterrows():
                        value = row["Value"]
                        if value is not None:
                            if "Charges" in feature:
                                formatted_value = f"${float(value):.2f}"
                            else:
                                formatted_value = f"{value}"
                            st.markdown(f"""
                            <div class="feature-card">
                                <strong>{feature}</strong><br>
                                {formatted_value}
                            </div>
                            """, unsafe_allow_html=True)

                # 📑 Contract
                with col3:
                    st.markdown("#### 📑 Contract")
                    for feature, row in contract_features.iterrows():
                        value = row["Value"]
                        if value is not None:
                            display_name = feature.replace("_", " ").title()
                            st.markdown(f"""
                            <div class="feature-card">
                                <strong>{display_name}</strong><br>
                                {value}
                            </div>
                            """, unsafe_allow_html=True)

            else:
                st.error(f"Failed to fetch features: {features_data.get('detail', 'Unknown error')}")

    # Pipeline Visualization
    st.markdown("---")
    st.subheader("🔗 End-to-End Pipeline Flow")
    
    # Create pipeline visualization
    pipeline_steps = [
        {"name": "1. Data Extraction", "status": "✅", "description": "Download from Kaggle"},
        {"name": "2. Feature Engineering", "status": "✅", "description": "Transform & create features"},
        {"name": "3. Feast Offline Store", "status": "✅", "description": "Store historical features"},
        {"name": "4. Model Training", "status": "✅", "description": "Train on point-in-time correct data"},
        {"name": "5. Feast Online Store", "status": "✅", "description": "Materialize for real-time serving"},
        {"name": "6. Real-time Inference", "status": "✅", "description": "Predict using fresh features"}
    ]
    
    for step in pipeline_steps:
        col1, col2, col3 = st.columns([1, 4, 8])
        with col1:
            st.success(step["status"])
        with col2:
            st.write(f"**{step['name']}**")
        with col3:
            st.write(step["description"])
    
    # Data Statistics
    st.markdown("---")
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(dashboard.customer_data))
    
    with col2:
        churn_rate = dashboard.customer_data['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_tenure = dashboard.customer_data['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_charges = dashboard.customer_data['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
    
    # Feature importance visualization (if available)
    try:
        with open("model/model_info.json", "r") as f:
            model_info = json.load(f)
        
        if 'feature_importance' in model_info:
            st.subheader("🎯 Feature Importance")
            
            # Get top 10 features
            feature_importance = model_info['feature_importance']
            importance_df = pd.DataFrame({
                'feature': list(feature_importance.keys()),
                'importance': list(feature_importance.values())
            }).sort_values('importance', ascending=True).tail(10)
            
            # Create horizontal bar chart
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='importance',
                color_continuous_scale='blues'
            )
            
            fig.update_layout(
                yaxis_title="Features",
                xaxis_title="Importance",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except:
        pass

if __name__ == "__main__":
    main()