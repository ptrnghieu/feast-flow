from feast import FeatureService
from features.customer_features_v2 import (
    customer_financials_fv_v2,
    customer_contract_fv_v2
)

churn_service_v2 = FeatureService(
    name="churn_service_v2",
    features=[
        customer_financials_fv_v2,
        customer_contract_fv_v2,
    ],
)