from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Define the primary entity
customer = Entity(
    name="customer",
    description="A telco customer",
    join_keys=["customerID"],
)

# Define the data source
customer_data_source = FileSource(
    name="customer_data_source",
    path=r"../data/processed/telco_churn_processed.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature views
customer_demographics_fv_v1 = FeatureView(
    name="customer_demographics_v1",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="gender", dtype=Int64),
        Field(name="SeniorCitizen", dtype=Int64),
        Field(name="Partner", dtype=Int64),
        Field(name="Dependents", dtype=Int64),
        Field(name="total_services", dtype=Int64),
    ],
    source=customer_data_source,
    online=True,
    tags={"team": "demographics"},
)

customer_financials_fv_v1 = FeatureView(
    name="customer_financials_v1",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="tenure", dtype=Int64),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
        Field(name="customer_tenure_ratio", dtype=Float32),
        Field(name="monthly_charge_avg", dtype=Float32),
    ],
    source=customer_data_source,
    online=True,
    tags={"team": "financials"},
)

customer_contract_fv_v1 = FeatureView(
    name="customer_contract_v1",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="InternetService_DSL", dtype=Int64),
        Field(name="InternetService_Fiber optic", dtype=Int64),
        Field(name="InternetService_No", dtype=Int64),
        Field(name="Contract_Month-to-month", dtype=Int64),
        Field(name="Contract_One year", dtype=Int64),
        Field(name="Contract_Two year", dtype=Int64),
        Field(name="PaymentMethod_Bank transfer (automatic)", dtype=Int64),
        Field(name="PaymentMethod_Credit card (automatic)", dtype=Int64),
        Field(name="PaymentMethod_Electronic check", dtype=Int64),
        Field(name="PaymentMethod_Mailed check", dtype=Int64),
    ],
    source=customer_data_source,
    online=True,
    tags={"team": "contract"},
)