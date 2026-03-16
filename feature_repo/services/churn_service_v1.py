from feast import FeatureService
from features.customer_features_v1 import (
    customer_demographics_fv_v1,
    customer_financials_fv_v1
)

churn_service_v1 = FeatureService(
    name="churn_service_v1",
    features=[
        customer_demographics_fv_v1,
        customer_financials_fv_v1,
    ],
)